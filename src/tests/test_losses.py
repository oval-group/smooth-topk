import torch
import unittest
import numpy as np

from torch.autograd import Variable
from losses.svm import SmoothTop1SVM, SmoothTopkSVM, MaxTop1SVM, MaxTopkSVM
from losses.functional import Topk_Smooth_SVM
from tests.utils import assert_all_close, V
from tests.py_ref import svm_topk_smooth_py_1, svm_topk_smooth_py_2,\
    smooth_svm_py, max_svm_py, svm_topk_max_py

from torch.autograd.gradcheck import gradcheck


class TestMaxSVM(unittest.TestCase):

    def setUp(self):

        torch.manual_seed(1234)
        np.random.seed(1234)

        self.n_samples = 20
        self.n_classes = 7
        self.alpha = 1.
        self.x = torch.randn(self.n_samples, self.n_classes)
        self.y = torch.from_numpy(np.random.randint(0, self.n_classes,
                                                    size=self.n_samples))
        self.k = 3

    def testMaxSVM(self):

        max_svm_th = MaxTop1SVM(self.n_classes, alpha=self.alpha)
        res_th = max_svm_th(V(self.x), V(self.y))
        res_py = max_svm_py(V(self.x), V(self.y), alpha=self.alpha)

        assert_all_close(res_th, res_py)

    def testMaxSVMtopk(self):

        max_svm_th = MaxTopkSVM(self.n_classes, k=self.k)
        res_th = max_svm_th(V(self.x), V(self.y))
        res_py = svm_topk_max_py(V(self.x), V(self.y), k=self.k)

        assert_all_close(res_th, res_py)


class TestSmoothSVM(unittest.TestCase):

    def setUp(self):

        torch.manual_seed(1234)
        np.random.seed(1234)

        self.n_samples = 20
        self.n_classes = 7
        self.tau = float(2.)
        self.x = torch.randn(self.n_samples, self.n_classes)
        self.y = torch.from_numpy(np.random.randint(0, self.n_classes,
                                                    size=self.n_samples))

    def testSmoothSVM(self):

        smooth_svm_th = SmoothTop1SVM(self.n_classes, tau=self.tau)
        res_th = smooth_svm_th(V(self.x), V(self.y))
        res_py = smooth_svm_py(V(self.x), V(self.y), self.tau)

        assert_all_close(res_th, res_py)


class TestSmoothSVMTopk(unittest.TestCase):

    def setUp(self):

        torch.manual_seed(1234)
        np.random.seed(1234)

        self.n_samples = 2
        self.n_classes = 7
        self.k = 5
        self.tau = float(2.)
        self.x = torch.randn(self.n_samples, self.n_classes)
        self.y = torch.from_numpy(np.random.randint(0, self.n_classes,
                                                    size=self.n_samples))
        self.labels = torch.from_numpy(np.arange(self.n_classes))

    def testSmoothSVMpy(self):

        res_py_1 = svm_topk_smooth_py_1(V(self.x), V(self.y), self.tau, self.k)
        res_py_2 = svm_topk_smooth_py_2(V(self.x), V(self.y), self.tau, self.k)

        assert_all_close(res_py_1, res_py_2)

    def testSmoothSVMth_functional(self):

        F = Topk_Smooth_SVM(self.labels, self.k, self.tau)
        res_th = F(V(self.x), V(self.y))
        res_py = svm_topk_smooth_py_1(V(self.x), V(self.y), self.tau, self.k)

        assert_all_close(res_th, res_py)

    def testSmoothSVMth_loss(self):

        svm_topk_smooth_th = SmoothTopkSVM(self.n_classes, tau=self.tau,
                                           k=self.k)
        res_th = svm_topk_smooth_th(V(self.x), V(self.y))
        res_py = svm_topk_smooth_py_1(V(self.x),
                                      V(self.y),
                                      self.tau, self.k).mean()

        assert_all_close(res_th, res_py)

    def testSmoothSVMth_loss_scales(self):

        svm_topk_smooth_th = SmoothTopkSVM(self.n_classes, tau=self.tau, k=self.k)
        for scale in (1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3):
            x = self.x * scale
            res_th = svm_topk_smooth_th(V(x), V(self.y))
            res_py = svm_topk_smooth_py_1(V(x), V(self.y), self.tau, self.k).mean()
            assert_all_close(res_th, res_py)

    def testGradSmoothSVMth_loss(self):

        svm_topk_smooth_th = SmoothTopkSVM(self.n_classes, tau=self.tau, k=self.k)
        for scale in (1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4):
            x = self.x * scale
            x = Variable(x, requires_grad=True)
            assert gradcheck(lambda x: svm_topk_smooth_th(x, V(self.y)),
                             (x,), atol=1e-2, rtol=1e-3, eps=max(1e-4 * scale, 1e-2)), \
                "failed with scale {}".format(scale)
