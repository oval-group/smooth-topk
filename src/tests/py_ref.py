import numpy as np
import scipy.misc as sp
import itertools

from tests.utils import to_numpy


def log1mexp_py(x):
    x = to_numpy(x).astype(np.float128)
    res = np.log(-np.expm1(x))
    return res


def max_svm_py(scores, y_truth, alpha=1.):

    scores = scores.data.numpy()
    y_truth = y_truth.data.numpy()

    objective = 0
    n_samples = scores.shape[0]
    n_classes = scores.shape[1]
    for i in range(n_samples):
        # find maximally violated constraint
        loss_augmented = np.array([scores[i, y] + alpha * int(y != y_truth[i])
                                  for y in range(n_classes)])
        y_star = np.argmax(loss_augmented)

        # update metrics
        delta = int(y_truth[i] != y_star) * alpha
        objective += max(delta + scores[i, y_star] - scores[i, y_truth[i]], 0)

    objective *= 1. / n_samples

    return objective


def svm_topk_max_py(scores, y_truth, k):

    assert k > 1

    scores = scores.data.numpy()
    y_truth = y_truth.data.numpy()

    objective = 0
    n_samples = scores.shape[0]
    n_classes = scores.shape[1]
    for i in range(n_samples):
        # all scores for sample i except ground truth score
        scores_ = np.array([scores[i, y] for y in range(n_classes)
                            if y != y_truth[i]])

        # k maximal scores excluding y_truth + loss of 1
        obj_1 = np.mean(np.sort(scores_)[-k:]) + 1.

        # k - 1 maximal scores excluding y_truth + score of y_truth
        obj_2 = (np.sum(np.sort(scores_)[-k + 1:]) + scores[i, y_truth[i]]) / k

        # update metrics
        objective += max(obj_1, obj_2) - obj_2

    objective *= 1. / n_samples

    return objective


def smooth_svm_py(x, y, tau):
    x, y = to_numpy(x), to_numpy(y)
    n_samples, n_classes = x.shape
    scores = x + np.not_equal(np.arange(n_classes)[None, :], y[:, None]) - \
        x[np.arange(n_samples), y][:, None]
    loss = tau * np.mean(sp.logsumexp(scores / tau, axis=1))
    return loss


def sum_product_py(x, k):
    x = to_numpy(x)
    n_samples, n_classes = x.shape
    res = np.zeros(n_samples)
    for indices in itertools.combinations(range(n_classes), k):
        res += np.product(x[:, indices], axis=1)
    return res


def svm_topk_smooth_py_1(x, y, tau, k):
    x, y = to_numpy(x), to_numpy(y)
    x = x.astype(np.float128)
    tau = float(tau)
    n_samples, n_classes = x.shape
    exp = np.exp(x * 1. / (k * tau))

    term_1 = np.zeros(n_samples)
    for indices in itertools.combinations(range(n_classes), k):
        delta = 1. - np.sum(indices == y[:, None], axis=1)
        term_1 += np.product(exp[:, indices], axis=1) * np.exp(delta / tau)

    term_2 = np.zeros(n_samples)
    for i in range(n_samples):
        all_but_y = [j for j in range(n_classes) if j != y[i]]
        for indices in itertools.combinations(all_but_y, k - 1):
            term_2[i] += np.product(exp[i, indices]) * exp[i, y[i]]

    loss = tau * (np.log(term_1) - np.log(term_2))

    return loss


def svm_topk_smooth_py_2(x, y, tau, k):
    x, y = to_numpy(x), to_numpy(y)
    n_samples, n_classes = x.shape
    exp = np.exp(x * 1. / (k * tau))

    term_1 = np.zeros(n_samples)
    for i in range(n_samples):
        all_but_y = [j for j in range(n_classes) if j != y[i]]
        for indices in itertools.combinations(all_but_y, k - 1):
            term_1[i] += np.product(exp[i, indices])

    term_2 = np.zeros(n_samples)
    for i in range(n_samples):
        all_but_y = [j for j in range(n_classes) if j != y[i]]
        for indices in itertools.combinations(all_but_y, k):
            term_2[i] += np.product(exp[i, indices])

    all_ = np.arange(n_samples)
    loss = tau * (np.log(term_1 * exp[all_, y] + np.exp(1. / tau) * term_2) -
                  np.log(term_1 * exp[all_, y]))
    return loss
