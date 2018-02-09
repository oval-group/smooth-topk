from losses.polynomial.multiplication import Multiplication
from losses.polynomial.divide_conquer import divide_and_conquer


def log_sum_exp_k(x, k):
    # number of samples and number of coefficients to compute
    n_s = x.size(0)

    assert k <= x.size(1)

    # clone to allow in-place operations
    x = x.clone()

    # pre-compute normalization
    x_summed = x.sum(1)

    # invert in log-space
    x.t_().mul_(-1)

    # initialize polynomials (in log-space)
    x = [x, x.clone().fill_(0)]

    # polynomial mulitplications
    log_res = divide_and_conquer(x, k, mul=Multiplication(k))

    # re-normalize
    coeff = log_res + x_summed[None, :]

    # avoid broadcasting issues (in particular if n_s = 1)
    coeff = coeff.view(k + 1, n_s)

    return coeff[k - 1: k + 1]
