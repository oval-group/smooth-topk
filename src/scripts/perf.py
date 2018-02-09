import argparse
import torch
import numpy as np
import itertools
import time

from torch.autograd import Variable

import sys
sys.path.insert(1, '.')

from losses.polynomial.sp import LogSumExp
from losses.utils import split
from tests.th_ref import log_sum_exp_k


def sum_k_pyref(x, k):
    exp = torch.exp(x.data.cpu().numpy())
    n_classes = x.shape[1]
    res = 1e-10 * np.ones(len(x))
    for indices in itertools.combinations(range(n_classes), k):
        res += np.product(exp[:, indices], axis=1)
    return res


def esf_py(x, k, buffer):
    xx = torch.exp(x)
    n = x.size(1)

    # use buffer below
    buffer.zero_()
    res = Variable(buffer)
    res[:, :-1, 0] = 1
    res[:, 0, 1] = xx[:, 0]

    xx = xx.unsqueeze(2)

    for i in range(1, n):
        m = max(1, i + k - n)
        M = min(i, k) + 1
        res[:, i, m:M] = res[:, i - 1, m:M] + \
            xx[:, i] * res[:, i - 1, m - 1: M - 1]

    return torch.log(res[:, -1, k - 1]), torch.log(res[:, -1, k])


parser = argparse.ArgumentParser()
parser.add_argument('--k', type=int, default=5)
parser.add_argument('--tau', type=float, default=0.1)
parser.add_argument('--n_classes', type=int, default=1000)
parser.add_argument('--batch-size', type=int, default=256)


args = parser.parse_args()
k = args.k
CUDA = torch.cuda.is_available()
tau = args.tau
batch_size = args.batch_size
n_classes = args.n_classes

print("=" * 90)
print("CONFIGURATION")
print("=" * 90)
print('-' * 90)
print('k: \t\t{}'.format(k))
print('C: \t\t{}'.format(n_classes))
print('Batch size: \t{}'.format(batch_size))
print('Tau: \t\t{}'.format(tau))
print('-' * 90)

torch.manual_seed(1234)

scores = Variable(torch.randn(batch_size, n_classes))
target = torch.from_numpy(np.random.randint(n_classes, size=batch_size))
labels = torch.from_numpy(np.arange(n_classes))

if CUDA:
    target = target.cuda()
    labels = labels.cuda()
    scores = scores.cuda()

x_1, x_2 = split(scores, Variable(target), labels)
x_1.div_(k * tau)
x_2.div_(k * tau)


def timing_fun(fun, x, k, verbosity, double=False, ntimes=50,
               forward=1, use_buffer=False):

    avg_clock = 0.
    for _ in range(ntimes):
        if double:
            x = x.double()
        x = Variable(x.data.clone(),
                     volatile=forward,
                     requires_grad=not forward)
        if use_buffer:
            buffer = x.data.new(x.size(0), x.size(1), k + 1)
        if CUDA:
            torch.cuda.synchronize()
        if forward:
            clock = -time.time()
        if use_buffer:
            skm1, sk = fun(x, k, buffer)
        else:
            skm1, sk = fun(x, k)
        if not forward:
            if CUDA:
                torch.cuda.synchronize()
            clock = -time.time()
            (skm1 + sk).sum().backward()
        if CUDA:
            torch.cuda.synchronize()
        clock += time.time()

        avg_clock += clock

        if verbosity:
            print(torch.stack((skm1.data, sk.data), dim=1).sum())

    # average over ntimes
    avg_clock /= ntimes

    return avg_clock, ntimes


def speed(verbosity=1, forward=1):

    print("=" * 90)
    print("SPEED")
    print("=" * 90)

    print('-' * 90)
    if forward:
        print('FORWARD')
    else:
        print('BACKWARD')
    print('-' * 90)

    if not forward:
        clock, ntimes = timing_fun(log_sum_exp_k, x_1, k, verbosity,
                                   double=False, forward=forward)
        print("Divide-and-conquer AD: \t{0:.3f}s / mini-batch \t (avg of {1} runs)".format(clock, ntimes))

    if forward:
        clock, ntimes = timing_fun(esf_py, x_1, k, verbosity,
                                   double=False, forward=forward,
                                   use_buffer=True)
        print("Summation Algorithm: \t{0:.3f}s / mini-batch \t (avg of {1} runs)".format(clock, ntimes))

    clock, ntimes = timing_fun(lambda x, y: LogSumExp(k)(x), x_1, k, verbosity,
                               double=False, forward=forward)
    print("Divide-and-conquer MD: \t{0:.3f}s / mini-batch \t (avg of {1} runs)".format(clock, ntimes))


def run_fun(fun, x, k, double=False, use_buffer=False):

    if double:
        x = x.double()
    x = Variable(x.data.clone(), volatile=True)

    if use_buffer:
        buffer = x.data.new(x.size(0), x.size(1), k + 1)
    if use_buffer:
        skm1, sk = fun(x, k, buffer)
    else:
        skm1, sk = fun(x, k)

    return skm1.data.cpu().numpy().sum() + sk.data.cpu().numpy().sum()


def stability():

    print("=" * 90)
    print("STABILITY")
    print("=" * 90)

    print('\n(Test successful if the number is not inf / nan)\n')

    res = run_fun(esf_py, x_1, k, double=False, use_buffer=True)
    print("Summation Algorithm (S): \t{}".format(res))

    res = run_fun(esf_py, x_1, k, double=True, use_buffer=True)
    print("Summation Algorithm (D): \t{}".format(res))

    res = run_fun(lambda x, y: LogSumExp(k)(x), x_1, k, double=False)
    print("Divide-and-conquer (S): \t{}".format(res))

    res = run_fun(lambda x, y: LogSumExp(k)(x), x_1, k, double=True)
    print("Divide-and-conquer (D): \t{}".format(res))


speed(verbosity=0, forward=1)
speed(verbosity=0, forward=0)

stability()
