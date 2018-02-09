import math
import torch
import unittest
import numpy as np

from losses.logarithm import LogTensor, log1mexp
from tests.utils import assert_all_close, V
from tests.py_ref import log1mexp_py


class TestLogTensor(unittest.TestCase):

    def setUp(self):

        torch.manual_seed(1234)

        self.n_element = 50
        self.x = torch.randn(self.n_element).abs()
        self.y = torch.randn(self.n_element).abs()
        self.nonzero_const = np.random.rand()

    def testSumTensors(self):

        sum_ = LogTensor(V(self.x)) + LogTensor(V(self.y))
        res_sb = sum_.torch()
        res_th = torch.log(torch.exp(self.x.double()) +
                           torch.exp(self.y.double()))

        assert_all_close(res_th, res_sb)

    def testSumNonZero(self):

        sum_ = LogTensor(V(self.x)) + self.nonzero_const
        res_sb = sum_.torch()
        res_th = torch.log(torch.exp(self.x.double()) +
                           self.nonzero_const)

        assert_all_close(res_th, res_sb)

    def testSumZero(self):

        sum_ = LogTensor(V(self.x)) + 0
        res_sb = sum_.torch()
        res_th = self.x

        assert_all_close(res_th, res_sb)

    def testMulTensors(self):

        sum_ = LogTensor(V(self.x)) * LogTensor(V(self.y))
        res_sb = sum_.torch()
        res_th = self.x.double() + self.y.double()

        assert_all_close(res_th, res_sb)

    def testMulNonZero(self):

        sum_ = LogTensor(V(self.x)) * self.nonzero_const
        res_sb = sum_.torch()
        res_th = self.x.double() + math.log(self.nonzero_const)

        assert_all_close(res_th, res_sb)

    def testMulZero(self):

        sum_ = LogTensor(V(self.x)) * 0
        res_sb = sum_.torch()
        res_th = -np.inf * np.ones(res_sb.size())

        assert_all_close(res_th, res_sb)


class Test1MExp(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(1234)
        shape = (100, 100)
        self.x = -torch.randn(shape).abs()


def gen_test_exp1m(scale):
    def test(cls):
        x = cls.x * 10 ** scale
        res_th = log1mexp(x)
        res_py = log1mexp_py(x)
        assert_all_close(res_th, res_py, rtol=1e-4, atol=1e-5)
    return test


def add_scale_tests_1mexp():
    for scale in (-3, -2, -1, 0, 1, 2, 3, 4):
        test = gen_test_exp1m(scale)
        test_name = 'test_scale_{}'.format(str(scale))
        setattr(Test1MExp, test_name, test)


add_scale_tests_1mexp()
