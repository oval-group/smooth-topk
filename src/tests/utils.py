import numpy as np
import torch

from torch.autograd import Variable


def V(x, requires_grad=False):
    """
    returns clone of tensor x wrapped in a Variable
    Avoids issue of inplace operations if x used in several functions
    """
    assert torch.is_tensor(x)
    return Variable(x.clone(), requires_grad=requires_grad)


def to_numpy(tensor):
    if isinstance(tensor, Variable):
        tensor = tensor.data
    if torch.is_tensor(tensor):
        tensor = tensor.clone().cpu().numpy()
    if not hasattr(tensor, '__len__'):
        tensor = np.array([tensor])
    assert isinstance(tensor, np.ndarray)
    tensor = tensor.squeeze()
    return tensor


def assert_all_close(tensor_1, tensor_2, rtol=1e-4, atol=1e-4):
    tensor_1 = to_numpy(tensor_1).astype(np.float64)
    tensor_2 = to_numpy(tensor_2).astype(np.float64)
    np.testing.assert_equal(np.isposinf(tensor_1),
                            np.isposinf(tensor_2))
    np.testing.assert_equal(np.isneginf(tensor_1),
                            np.isneginf(tensor_2))
    indices = np.isfinite(tensor_1)
    if indices.sum():
        tensor_1 = tensor_1[indices]
        tensor_2 = tensor_2[indices]
        err = np.max(np.abs(tensor_1 - tensor_2))
        err_msg = "Max abs error: {0:.3g}".format(err)
        np.testing.assert_allclose(tensor_1, tensor_2, rtol=rtol, atol=atol,
                                   err_msg=err_msg)
