import math
import torch
import unittest
import numpy as np

from losses.polynomial import LogSumExp
from tests.utils import assert_all_close, V
from tests.py_ref import sum_product_py
from tests.th_ref import log_sum_exp_k
from torch.autograd import Variable, gradcheck


class TestSumProduct(unittest.TestCase):

    def setUp(self):

        torch.set_printoptions(linewidth=160, threshold=1e3)

        seed = 7
        np.random.seed(1234)
        seed = np.random.randint(1e5)
        torch.manual_seed(seed)

        self.eps = 1e-4

    def testLogSumProductExp(self):

        self.n_samples = 25
        self.n_classes = 20
        self.k = 7
        self.x = torch.randn(self.n_samples, self.n_classes)

        res_th = LogSumExp(self.k, p=1)(V(self.x)).squeeze()
        res1_th, res2_th = res_th[0], res_th[1]
        res1_py = np.log(sum_product_py(V(torch.exp(self.x)), self.k - 1))
        res2_py = np.log(sum_product_py(V(torch.exp(self.x)), self.k))

        assert_all_close(res1_th, res1_py)
        assert_all_close(res2_th, res2_py)

    def test_backward(self):

        self.n_samples = 25
        self.n_classes = 1000
        self.k = 100
        self.k = 20
        self.x = torch.randn(self.n_samples, self.n_classes)
        self.x, _ = torch.sort(self.x, dim=1, descending=True)

        for tau in (5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1, 5, 1e1, 5e2, 1e3):
            x = self.x / (tau * self.k)
            top, _ = x.topk(self.k + 1, 1)
            thresh = 1e2
            hard = torch.ge(top[:, self.k - 1] - top[:, self.k],
                            math.log(thresh))
            smooth = hard.eq(0)

            x = x[smooth.unsqueeze(1).expand_as(x)].view(-1, x.size(1))
            if not x.size():
                print('empty tensor')
                return

            X_auto = Variable(x.double(), requires_grad=True)
            X_man = Variable(x, requires_grad=True)

            res1_auto, res2_auto = log_sum_exp_k(X_auto, self.k)
            res1_auto, res2_auto = res1_auto.squeeze(), res2_auto.squeeze()

            res_man = LogSumExp(self.k)(X_man).squeeze()
            res1_man = res_man[0]
            res2_man = res_man[1]

            proj1 = torch.randn(res1_auto.size()).fill_(1)
            proj2 = torch.randn(res2_auto.size()).fill_(1)

            proj_auto = torch.dot(V(proj1.double()), res1_auto) +\
                torch.dot(V(proj2.double()), res2_auto)
            proj_man = torch.dot(V(proj1), res1_man) +\
                torch.dot(V(proj2), res2_man)
            proj_auto.backward()
            proj_man.backward()

            # check forward
            assert_all_close(res1_auto, res1_man, atol=1e0, rtol=1e-3)
            assert_all_close(res2_auto, res2_man, atol=1e0, rtol=1e-3)

            # check backward
            assert_all_close(X_auto.grad, X_man.grad, atol=0.05, rtol=1e-2)