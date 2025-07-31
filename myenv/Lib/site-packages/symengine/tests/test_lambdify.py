import array
import cmath
from functools import reduce
import itertools
from operator import mul
import math

import symengine as se
from symengine.test_utilities import raises
from symengine import have_numpy
import unittest
from unittest.case import SkipTest

try:
    import sympy
    from sympy.core.cache import clear_cache
    import atexit
    atexit.register(clear_cache)
    have_sympy = True
except ImportError:
    have_sympy = False

try:
    import scipy
    from scipy import LowLevelCallable
    have_scipy = True
except ImportError:
    have_scipy = False

if have_numpy:
    import numpy as np

def _size(arr):
    try:
        return arr.memview.size
    except AttributeError:
        return len(arr)


def isclose(a, b, rtol=1e-13, atol=1e-13):
    discr = a - b
    toler = (atol + rtol*abs(a))
    return abs(discr) < toler


def allclose(vec1, vec2, rtol=1e-13, atol=1e-13):
    n1, n2 = _size(vec1), _size(vec2)
    if n1 != n2:
        return False

    for idx in range(n1):
        if not isclose(vec1[idx], vec2[idx], rtol, atol):
            return False
    return True


@unittest.skipUnless(have_numpy, "Numpy not installed")
def test_ravel():
    x = se.symbols('x')
    exprs = [x+1, x+2, x+3, 1/x, 1/(x*x), 1/(x**3.0)]
    A = se.DenseMatrix(2, 3, exprs)
    assert np.all(np.ravel(A, order='C') == exprs)


@unittest.skipUnless(have_numpy, "Numpy not installed")
def test_Lambdify():
    n = 7
    args = x, y, z = se.symbols('x y z')
    L = se.Lambdify(args, [x+y+z, x**2, (x-y)/z, x*y*z], backend='lambda')
    assert allclose(L(range(n, n+len(args))),
                    [3*n+3, n**2, -1/(n+2), n*(n+1)*(n+2)])

@unittest.skipUnless(have_numpy, "Numpy not installed")
def test_Lambdify_with_opt_level():
    args = x, y, z = se.symbols('x y z')
    raises(TypeError, lambda: se.Lambdify(args, [x+y+z, x**2, (x-y)/z, x*y*z], backend='lambda', opt_level=0))

def _test_Lambdify_Piecewise(Lambdify):
    x = se.symbols('x')
    p = se.Piecewise((-x, x<0), (x*x*x, True))
    f = Lambdify([x], [p])
    arr = np.linspace(3, 7)
    assert np.allclose(f(-arr).flat, arr, atol=1e-14, rtol=1e-15)
    assert np.allclose(f(arr).flat, arr**3, atol=1e-14, rtol=1e-15)


@unittest.skipUnless(have_numpy, "Numpy not installed")
def test_Lambdify_Piecewise():
    _test_Lambdify_Piecewise(lambda *args: se.Lambdify(*args, backend='lambda'))
    if se.have_llvm:
        _test_Lambdify_Piecewise(lambda *args: se.Lambdify(*args, backend='llvm'))

@unittest.skipUnless(have_numpy, "Numpy not installed")
def test_Lambdify_LLVM():
    n = 7
    args = x, y, z = se.symbols('x y z')
    if not se.have_llvm:
        raises(ValueError, lambda: se.Lambdify(args, [x+y+z, x**2,
                                                      (x-y)/z, x*y*z],
                                               backend='llvm'))
        raise SkipTest("No LLVM support")
    L = se.Lambdify(args, [x+y+z, x**2, (x-y)/z, x*y*z], backend='llvm')
    assert allclose(L(range(n, n+len(args))),
                    [3*n+3, n**2, -1/(n+2), n*(n+1)*(n+2)])

@unittest.skipUnless(have_numpy, "Numpy not installed")
def test_Lambdify_LLVM_with_opt_level():
    for opt_level in range(4):
        n = 7
        args = x, y, z = se.symbols('x y z')
        if not se.have_llvm:
            raises(ValueError, lambda: se.Lambdify(args, [x+y+z, x**2,
                                                          (x-y)/z, x*y*z],
                                                   backend='llvm', opt_level=opt_level))
            raise SkipTest("No LLVM support")
        L = se.Lambdify(args, [x+y+z, x**2, (x-y)/z, x*y*z], backend='llvm', opt_level=opt_level)
        assert allclose(L(range(n, n+len(args))),
                        [3*n+3, n**2, -1/(n+2), n*(n+1)*(n+2)])

def _get_2_to_2by2():
    args = x, y = se.symbols('x y')
    exprs = np.array([[x+y+1.0, x*y],
                      [x/y, x**y]])
    L = se.Lambdify(args, exprs)

    def check(A, inp):
        X, Y = inp
        assert abs(A[0, 0] - (X+Y+1.0)) < 1e-15
        assert abs(A[0, 1] - (X*Y)) < 1e-15
        assert abs(A[1, 0] - (X/Y)) < 1e-15
        assert abs(A[1, 1] - (X**Y)) < 1e-13
    return L, check


@unittest.skipUnless(have_numpy, "Numpy not installed")
def test_Lambdify_2dim():
    lmb, check = _get_2_to_2by2()
    for inp in [(5, 7), np.array([5, 7]), [5.0, 7.0]]:
        A = lmb(inp)
        assert A.shape == (2, 2)
        check(A, inp)


def _get_array():
    X, Y, Z = inp = array.array('d', [1, 2, 3])
    args = x, y, z = se.symbols('x y z')
    exprs = [x+y+z, se.sin(x)*se.log(y)*se.exp(z)]
    ref = [X+Y+Z, math.sin(X)*math.log(Y)*math.exp(Z)]

    def check(arr):
        assert all([abs(x1-x2) < 1e-13 for x1, x2 in zip(ref, arr)])
    return args, exprs, inp, check


@unittest.skipUnless(have_numpy, "Numpy not installed")
def test_array():
    args, exprs, inp, check = _get_array()
    lmb = se.Lambdify(args, exprs)
    out = lmb(inp)
    check(out)


@unittest.skipUnless(have_numpy, "Numpy not installed")
def test_numpy_array_out_exceptions():
    args, exprs, inp, check = _get_array()
    assert len(args) == 3 and len(exprs) == 2
    lmb = se.Lambdify(args, exprs)

    all_right = np.empty(len(exprs))
    lmb(inp, out=all_right)

    too_short = np.empty(len(exprs) - 1)
    raises(ValueError, lambda: (lmb(inp, out=too_short)))

    wrong_dtype = np.empty(len(exprs), dtype=int)
    raises(ValueError, lambda: (lmb(inp, out=wrong_dtype)))

    read_only = np.empty(len(exprs))
    read_only.flags['WRITEABLE'] = False
    raises(ValueError, lambda: (lmb(inp, out=read_only)))

    all_right_broadcast_C = np.empty((4, len(exprs)), order='C')
    inp_bcast = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    lmb(np.array(inp_bcast), out=all_right_broadcast_C)

    noncontig_broadcast = np.empty((4, len(exprs), 3)).transpose((1, 2, 0))
    raises(ValueError, lambda: (lmb(inp_bcast, out=noncontig_broadcast)))

    all_right_broadcast_F = np.empty((len(exprs), 4), order='F')
    lmb.order = 'F'
    lmb(np.array(np.array(inp_bcast).T), out=all_right_broadcast_F)



@unittest.skipUnless(have_numpy, "Numpy not installed")
def test_broadcast():
    a = np.linspace(-np.pi, np.pi)
    inp = np.ascontiguousarray(np.vstack((np.cos(a), np.sin(a))).T)  # 50 rows 2 cols
    assert inp.flags['C_CONTIGUOUS']
    x, y = se.symbols('x y')
    distance = se.Lambdify([x, y], [se.sqrt(x**2 + y**2)])
    assert np.allclose(distance([inp[0, 0], inp[0, 1]]), [1])
    dists = distance(inp)
    assert dists.shape == (50, 1)
    assert np.allclose(dists, 1)


@unittest.skipUnless(have_numpy, "Numpy not installed")
def test_broadcast_multiple_extra_dimensions():
    inp = np.arange(12.).reshape((4, 3, 1))
    x = se.symbols('x')
    cb = se.Lambdify([x], [x**2, x**3])
    assert np.allclose(cb([inp[0, 2]]), [4, 8])
    out = cb(inp)
    assert out.shape == (4, 3, 1, 2)
    out = out.squeeze()
    assert abs(out[2, 1, 0] - 7**2) < 1e-14
    assert abs(out[2, 1, 1] - 7**3) < 1e-14
    assert abs(out[-1, -1, 0] - 11**2) < 1e-14
    assert abs(out[-1, -1, 1] - 11**3) < 1e-14


def _get_cse_exprs():
    args = x, y = se.symbols('x y')
    exprs = [x*x + y, y/(x*x), y*x*x+x]
    inp = [11, 13]
    ref = [121+13, 13/121, 13*121 + 11]
    return args, exprs, inp, ref


@unittest.skipUnless(have_numpy, "Numpy not installed")
def test_cse():
    args, exprs, inp, ref = _get_cse_exprs()
    lmb = se.Lambdify(args, exprs, cse=True)
    out = lmb(inp)
    assert allclose(out, ref)


@unittest.skipUnless(have_numpy, "Numpy not installed")
def test_cse_gh174():
    x = se.symbols('x')
    funcs = [se.cos(x)**i for i in range(5)]
    f_lmb = se.Lambdify([x], funcs)
    f_cse = se.Lambdify([x], funcs, cse=True)
    a = np.array([1, 2, 3])
    assert np.allclose(f_lmb(a), f_cse(a))


def _get_cse_exprs_big():
    # this is essentially a performance test (can be replaced by a benchmark)
    x, p = se.symarray('x', 14), se.symarray('p', 14)
    exp = se.exp
    exprs = [
        x[0] + x[1] - x[4] + 36.252574322669, x[0] - x[2] + x[3] + 21.3219379611249,
        x[3] + x[5] - x[6] + 9.9011158998744, 2*x[3] + x[5] - x[7] + 18.190422234653,
        3*x[3] + x[5] - x[8] + 24.8679190043357, 4*x[3] + x[5] - x[9] + 29.9336062089226,
        -x[10] + 5*x[3] + x[5] + 28.5520551531262, 2*x[0] + x[11] - 2*x[4] - 2*x[5] + 32.4401680272417,
        3*x[1] - x[12] + x[5] + 34.9992934135095, 4*x[1] - x[13] + x[5] + 37.0716199972041,
        (p[0] - p[1] + 2*p[10] + 2*p[11] - p[12] - 2*p[13] + p[2] + 2*p[5] + 2*p[6] + 2*p[7] +
         2*p[8] + 2*p[9] - exp(x[0]) + exp(x[1]) - 2*exp(x[10]) - 2*exp(x[11]) + exp(x[12]) +
         2*exp(x[13]) - exp(x[2]) - 2*exp(x[5]) - 2*exp(x[6]) - 2*exp(x[7]) - 2*exp(x[8]) - 2*exp(x[9])),
        (-p[0] - p[1] - 15*p[10] - 2*p[11] - 3*p[12] - 4*p[13] - 4*p[2] - 3*p[3] - 2*p[4] - 3*p[6] -
         6*p[7] - 9*p[8] - 12*p[9] + exp(x[0]) + exp(x[1]) + 15*exp(x[10]) + 2*exp(x[11]) +
         3*exp(x[12]) + 4*exp(x[13]) + 4*exp(x[2]) + 3*exp(x[3]) + 2*exp(x[4]) + 3*exp(x[6]) +
         6*exp(x[7]) + 9*exp(x[8]) + 12*exp(x[9])),
        (-5*p[10] - p[2] - p[3] - p[6] - 2*p[7] - 3*p[8] - 4*p[9] + 5*exp(x[10]) + exp(x[2]) + exp(x[3]) +
         exp(x[6]) + 2*exp(x[7]) + 3*exp(x[8]) + 4*exp(x[9])),
        -p[1] - 2*p[11] - 3*p[12] - 4*p[13] - p[4] + exp(x[1]) + 2*exp(x[11]) + 3*exp(x[12]) + 4*exp(x[13]) + exp(x[4]),
        (-p[10] - 2*p[11] - p[12] - p[13] - p[5] - p[6] - p[7] - p[8] - p[9] + exp(x[10]) +
         2*exp(x[11]) + exp(x[12]) + exp(x[13]) + exp(x[5]) + exp(x[6]) + exp(x[7]) + exp(x[8]) + exp(x[9]))
    ]
    return tuple(x) + tuple(p), exprs, np.ones(len(x) + len(p))


@unittest.skipUnless(have_numpy, "Numpy not installed")
def test_cse_big():
    args, exprs, inp = _get_cse_exprs_big()
    lmb = se.Lambdify(args, exprs, cse=True)
    out = lmb(inp)
    ref = [expr.xreplace(dict(zip(args, inp))) for expr in exprs]
    assert allclose(out, ref)


@unittest.skipUnless(have_numpy, "Numpy not installed")
def test_broadcast_c():
    n = 3
    inp = np.arange(2*n).reshape((n, 2))
    assert inp.flags['C_CONTIGUOUS']
    lmb, check = _get_2_to_2by2()
    A = lmb(inp)
    assert A.shape == (3, 2, 2)
    for i in range(n):
        check(A[i, ...], inp[i, :])


@unittest.skipUnless(have_numpy, "Numpy not installed")
def test_broadcast_fortran():
    n = 3
    inp = np.arange(2*n).reshape((n, 2), order='F')
    lmb, check = _get_2_to_2by2()
    A = lmb(inp)
    assert A.shape == (3, 2, 2)
    for i in range(n):
        check(A[i, ...], inp[i, :])


def _get_1_to_2by3_matrix(Mtx=se.DenseMatrix):
    x = se.symbols('x')
    args = x,
    exprs = Mtx(2, 3, [x+1, x+2, x+3,
                       1/x, 1/(x*x), 1/(x**3.0)])
    L = se.Lambdify(args, exprs)

    def check(A, inp):
        X, = inp
        assert abs(A[0, 0] - (X+1)) < 1e-15
        assert abs(A[0, 1] - (X+2)) < 1e-15
        assert abs(A[0, 2] - (X+3)) < 1e-15
        assert abs(A[1, 0] - (1/X)) < 1e-15
        assert abs(A[1, 1] - (1/(X*X))) < 1e-15
        assert abs(A[1, 2] - (1/(X**3.0))) < 1e-15
    return L, check


@unittest.skipUnless(have_numpy, "Numpy not installed")
def test_2dim_Matrix():
    L, check = _get_1_to_2by3_matrix()
    inp = [7]
    check(L(inp), inp)


@unittest.skipUnless(have_numpy, "Numpy not installed")
@unittest.skipUnless(have_sympy, "SymPy not installed")
def test_2dim_Matrix__sympy():
    import sympy as sp
    L, check = _get_1_to_2by3_matrix(sp.Matrix)
    inp = [7]
    check(L(inp), inp)



def _test_2dim_Matrix_broadcast():
    L, check = _get_1_to_2by3_matrix()
    inp = range(1, 5)
    out = L(inp)
    for i in range(len(inp)):
        check(out[i, ...], (inp[i],))



@unittest.skipUnless(have_numpy, "Numpy not installed")
def test_2dim_Matrix_broadcast():
    _test_2dim_Matrix_broadcast()


@unittest.skipUnless(have_numpy, "Numpy not installed")
def test_2dim_Matrix_broadcast_multiple_extra_dim():
    L, check = _get_1_to_2by3_matrix()
    inp = np.arange(1, 4*5*6+1).reshape((4, 5, 6))
    out = L(inp)
    assert out.shape == (4, 5, 6, 2, 3)
    for i, j, k in itertools.product(range(4), range(5), range(6)):
        check(out[i, j, k, ...], (inp[i, j, k],))


@unittest.skipUnless(have_numpy, "Numpy not installed")
def test_jacobian():
    x, y = se.symbols('x, y')
    args = se.DenseMatrix(2, 1, [x, y])
    v = se.DenseMatrix(2, 1, [x**3 * y, (x+1)*(y+1)])
    jac = v.jacobian(args)
    lmb = se.Lambdify(args, jac)
    out = np.empty((2, 2))
    inp = X, Y = 7, 11
    lmb(inp, out=out)
    assert np.allclose(out, [[3 * X**2 * Y, X**3],
                             [Y + 1, X + 1]])


@unittest.skipUnless(have_numpy, "Numpy not installed")
def test_jacobian__broadcast():
    x, y = se.symbols('x, y')
    args = se.DenseMatrix(2, 1, [x, y])
    v = se.DenseMatrix(2, 1, [x**3 * y, (x+1)*(y+1)])
    jac = v.jacobian(args)
    lmb = se.Lambdify(args, jac)
    out = np.empty((3, 2, 2))
    inp0 = 7, 11
    inp1 = 8, 13
    inp2 = 5, 9
    inp = np.array([inp0, inp1, inp2])
    lmb(inp, out=out)
    for idx, (X, Y) in enumerate([inp0, inp1, inp2]):
        assert np.allclose(out[idx, ...], [[3 * X**2 * Y, X**3],
                                           [Y + 1, X + 1]])


@unittest.skipUnless(have_numpy, "Numpy not installed")
def test_excessive_args():
    x = se.symbols('x')
    lmb = se.Lambdify([x], [-x])
    inp = np.ones(2)
    out = lmb(inp)
    assert np.allclose(inp, [1, 1])
    assert len(out) == 2  # broad casting
    assert np.allclose(out, -1)


@unittest.skipUnless(have_numpy, "Numpy not installed")
def test_excessive_out():
    x = se.symbols('x')
    lmb = se.Lambdify([x], [-x])
    inp = np.ones(1)
    out = np.ones(2)
    _ = lmb(inp, out=out[:inp.size])
    assert np.allclose(inp, [1, 1])
    assert out[0] == -1
    assert out[1] == 1


def all_indices(shape):
    return itertools.product(*(range(dim) for dim in shape))


def ravelled(A):
    try:
        return A.ravel()
    except AttributeError:
        L = []
        for idx in all_indices(A.memview.shape):
            L.append(A[idx])
        return L


def _get_2_to_2by2_list(real=True):
    args = x, y = se.symbols('x y')
    exprs = [[x + y*y, y*y], [x*y*y, se.sqrt(x)+y*y]]
    L = se.Lambdify(args, exprs, real=real)

    def check(A, inp):
        X, Y = inp
        assert A.shape[-2:] == (2, 2)
        ref = [X + Y*Y, Y*Y, X*Y*Y, cmath.sqrt(X)+Y*Y]
        ravA = ravelled(A)
        size = _size(ravA)
        for i in range(size//4):
            for j in range(4):
                assert isclose(ravA[i*4 + j], ref[j])
    return L, check


@unittest.skipUnless(have_numpy, "Numpy not installed")
def test_2_to_2by2():
    L, check = _get_2_to_2by2_list()
    inp = [13, 17]
    A = L(inp)
    check(A, inp)


@unittest.skipUnless(have_numpy, "Numpy not installed")
def test_unsafe_real():
    L, check = _get_2_to_2by2_list()
    inp = np.array([13., 17.])
    out = np.empty(4)
    L.unsafe_real(inp, out)
    check(out.reshape((2, 2)), inp)


@unittest.skipUnless(have_numpy, "Numpy not installed")
def test_unsafe_complex():
    L, check = _get_2_to_2by2_list(real=False)
    assert not L.real
    inp = np.array([13+11j, 7+4j], dtype=np.complex128)
    out = np.empty(4, dtype=np.complex128)
    L.unsafe_complex(inp, out)
    check(out.reshape((2, 2)), inp)


@unittest.skipUnless(have_numpy, "Numpy not installed")
def test_itertools_chain():
    args, exprs, inp, check = _get_array()
    L = se.Lambdify(args, exprs)
    inp = itertools.chain([inp[0]], (inp[1],), [inp[2]])
    A = L(inp)
    check(A)


@unittest.skipUnless(have_numpy, "Numpy not installed")
def test_complex_1():
    x = se.Symbol('x')
    lmb = se.Lambdify([x], [1j + x], real=False)
    assert abs(lmb([11+13j])[0] -
               (11 + 14j)) < 1e-15


@unittest.skipUnless(have_numpy, "Numpy not installed")
def test_complex_2():
    x = se.Symbol('x')
    lmb = se.Lambdify([x], [3 + x - 1j], real=False)
    assert abs(lmb([11+13j])[0] -
               (14 + 12j)) < 1e-15


@unittest.skipUnless(have_numpy, "Numpy not installed")
def test_more_than_255_args():
    # SymPy's lambdify can handle at most 255 arguments
    # this is a proof of concept that this limitation does
    # not affect SymEngine's Lambdify class
    n = 257
    x = se.symarray('x', n)
    p, q, r = 17, 42, 13
    terms = [i*s for i, s in enumerate(x, p)]
    exprs = [se.add(*terms), r + x[0], -99]
    callback = se.Lambdify(x, exprs)
    input_arr = np.arange(q, q + n*n).reshape((n, n))
    out = callback(input_arr)
    ref = np.empty((n, 3))
    coeffs = np.arange(p, p + n, dtype=np.int64)
    for i in range(n):
        ref[i, 0] = coeffs.dot(np.arange(q + n*i, q + n*(i+1), dtype=np.int64))
        ref[i, 1] = q + n*i + r
    ref[:, 2] = -99
    assert np.allclose(out, ref)


def _Lambdify_heterogeneous_output(Lambdify):
    x, y = se.symbols('x, y')
    args = se.DenseMatrix(2, 1, [x, y])
    v = se.DenseMatrix(2, 1, [x**3 * y, (x+1)*(y+1)])
    jac = v.jacobian(args)
    exprs = [jac, x+y, v, (x+1)*(y+1)]
    lmb = Lambdify(args, *exprs)
    inp0 = 7, 11
    inp1 = 8, 13
    inp2 = 5, 9
    inp = np.array([inp0, inp1, inp2])
    o_j, o_xpy, o_v, o_xty = lmb(inp)
    for idx, (X, Y) in enumerate([inp0, inp1, inp2]):
        assert np.allclose(o_j[idx, ...], [[3 * X**2 * Y, X**3],
                                           [Y + 1, X + 1]])
        assert np.allclose(o_xpy[idx, ...], [X+Y])
        assert np.allclose(o_v[idx, ...], [[X**3 * Y], [(X+1)*(Y+1)]])
        assert np.allclose(o_xty[idx, ...], [(X+1)*(Y+1)])


@unittest.skipUnless(have_numpy, "Numpy not installed")
def test_Lambdify_heterogeneous_output():
    _Lambdify_heterogeneous_output(se.Lambdify)


def _sympy_lambdify_heterogeneous_output(cb, Mtx):
    x, y = se.symbols('x, y')
    args = Mtx(2, 1, [x, y])
    v = Mtx(2, 1, [x**3 * y, (x+1)*(y+1)])
    jac = v.jacobian(args)
    exprs = [jac, x+y, v, (x+1)*(y+1)]
    lmb = cb(args, exprs)
    inp0 = 7, 11
    inp1 = 8, 13
    inp2 = 5, 9
    for idx, (X, Y) in enumerate([inp0, inp1, inp2]):
        o_j, o_xpy, o_v, o_xty = lmb(X, Y)
        assert np.allclose(o_j, [[3 * X**2 * Y, X**3],
                                 [Y + 1, X + 1]])
        assert np.allclose(o_xpy, [X+Y])
        assert np.allclose(o_v, [[X**3 * Y], [(X+1)*(Y+1)]])
        assert np.allclose(o_xty, [(X+1)*(Y+1)])


@unittest.skipUnless(have_numpy, "Numpy not installed")
@unittest.skipUnless(have_sympy, "SymPy not installed")
def test_lambdify__sympy():
    import sympy as sp
    _sympy_lambdify_heterogeneous_output(se.lambdify, se.DenseMatrix)
    _sympy_lambdify_heterogeneous_output(sp.lambdify, sp.Matrix)


def _test_Lambdify_scalar_vector_matrix(Lambdify):
    if not have_numpy:
        return
    args = x, y = se.symbols('x y')
    vec = se.DenseMatrix([x+y, x*y])
    jac = vec.jacobian(se.DenseMatrix(args))
    f = Lambdify(args, x**y, vec, jac)
    assert f.n_exprs == 3
    s, v, m = f([2, 3])
    assert s == 2**3
    assert np.allclose(v, [[2+3], [2*3]])
    assert np.allclose(m, [
        [1, 1],
        [3, 2]
    ])

    for inp in [[2, 3, 5, 7], np.array([[2, 3], [5, 7]])]:
        s2, v2, m2 = f(inp)
        assert np.allclose(s2, [2**3, 5**7])
        assert np.allclose(v2, [
            [[2+3], [2*3]],
            [[5+7], [5*7]]
        ])
        assert np.allclose(m2, [
            [
                [1, 1],
                [3, 2]
            ],
            [
                [1, 1],
                [7, 5]
            ]
        ])


def test_Lambdify_scalar_vector_matrix():
    _test_Lambdify_scalar_vector_matrix(lambda *args: se.Lambdify(*args, backend='lambda'))
    if se.have_llvm:
        _test_Lambdify_scalar_vector_matrix(lambda *args: se.Lambdify(*args, backend='llvm'))


def test_Lambdify_scalar_vector_matrix_cse():
    _test_Lambdify_scalar_vector_matrix(lambda *args: se.Lambdify(*args, backend='lambda', cse=True))
    if se.have_llvm:
        _test_Lambdify_scalar_vector_matrix(lambda *args: se.Lambdify(*args, backend='llvm', cse=True))


@unittest.skipUnless(have_numpy, "Numpy not installed")
def test_Lambdify_gh174():
    # Tests array broadcasting if the expressions form an N-dimensional array
    # of say shape (k, l, m) and it contains 'n' arguments (x1, ... xn), then
    # if the user provides a Fortran ordered (column-major) input array of shape
    # (n, o, p, q), then the returned array will be of shape (k, l, m, o, p, q)
    args = x, y = se.symbols('x y')
    nargs = len(args)
    vec1 = se.DenseMatrix([x, x**2, x**3])
    assert vec1.shape == (3, 1)
    assert np.asarray(vec1).shape == (3, 1)
    lmb1 = se.Lambdify([x], vec1)
    out1 = lmb1(3)
    assert out1.shape == (3, 1)
    assert np.all(out1 == [[3], [9], [27]])
    assert lmb1([2, 3]).shape == (2, 3, 1)
    lmb1.order = 'F'  # change order
    out1a = lmb1([2, 3])
    assert out1a.shape == (3, 1, 2)
    ref1a_squeeze = [[2, 3],
                     [4, 9],
                     [8, 27]]
    assert np.all(out1a.squeeze() == ref1a_squeeze)
    assert out1a.flags['F_CONTIGUOUS']
    assert not out1a.flags['C_CONTIGUOUS']

    lmb2c = se.Lambdify(args, vec1, x+y, order='C')
    lmb2f = se.Lambdify(args, vec1, x+y, order='F')
    for out2a in [lmb2c([2, 3]), lmb2f([2, 3])]:
        assert np.all(out2a[0] == [[2], [4], [8]])
        assert out2a[0].ndim == 2
        assert out2a[1] == 5
        assert out2a[1].ndim == 0
    inp2b = np.array([
        [2.0, 3.0],
        [1.0, 2.0],
        [0.0, 6.0]
    ])
    raises(ValueError, lambda: (lmb2c(inp2b.T)))
    out2c = lmb2c(inp2b)
    out2f = lmb2f(np.asfortranarray(inp2b.T))
    assert out2c[0].shape == (3, 3, 1)
    assert out2f[0].shape == (3, 1, 3)
    for idx, (_x, _y) in enumerate(inp2b):
        assert np.all(out2c[0][idx, ...] == [[_x], [_x**2], [_x**3]])

    assert np.all(out2c[1] == [5, 3, 6])
    assert np.all(out2f[1] == [5, 3, 6])
    assert out2c[1].shape == (3,)
    assert out2f[1].shape == (3,)

    def _mtx3(_x, _y):
        return [[_x**row_idx + _y**col_idx for col_idx in range(3)]
                for row_idx in range(4)]
    mtx3c = np.array(_mtx3(x, y), order='C')
    mtx3f = np.array(_mtx3(x, y), order='F')
    lmb3c = se.Lambdify([x, y], x*y, mtx3c, vec1, order='C')
    lmb3f = se.Lambdify([x, y], x*y, mtx3f, vec1, order='F')
    inp3c = np.array([[2., 3], [3, 4], [5, 7], [6, 2], [3, 1]])
    inp3f = np.asfortranarray(inp3c.T)
    raises(ValueError, lambda: (lmb3c(inp3c.T)))
    out3c = lmb3c(inp3c)
    assert out3c[0].shape == (5,)
    assert out3c[1].shape == (5, 4, 3)
    assert out3c[2].shape == (5, 3, 1)  # user can apply numpy.squeeze if they want to.
    for a, b in zip(out3c, lmb3c(np.ravel(inp3c))):
        assert np.all(a == b)

    out3f = lmb3f(inp3f)
    assert out3f[0].shape == (5,)
    assert out3f[1].shape == (4, 3, 5)
    assert out3f[2].shape == (3, 1, 5)  # user can apply numpy.squeeze if they want to.
    for a, b in zip(out3f, lmb3f(np.ravel(inp3f, order='F'))):
        assert np.all(a == b)

    for idx, (_x, _y) in enumerate(inp3c):
        assert out3c[0][idx] == _x*_y
        assert out3f[0][idx] == _x*_y
        assert np.all(out3c[1][idx, ...] == _mtx3(_x, _y))
        assert np.all(out3f[1][..., idx] == _mtx3(_x, _y))
        assert np.all(out3c[2][idx, ...] == [[_x],[_x**2],[_x**3]])
        assert np.all(out3f[2][..., idx] == [[_x],[_x**2],[_x**3]])


def _get_Ndim_args_exprs_funcs(order):
    args = x, y = se.symbols('x y')

    # Higher dimensional inputs
    def f_a(index, _x, _y):
        a, b, c, d = index
        return _x**a + _y**b + (_x+_y)**-d

    nd_exprs_a = np.zeros((3, 5, 1, 4), dtype=object, order=order)
    for index in np.ndindex(*nd_exprs_a.shape):
        nd_exprs_a[index] = f_a(index, x, y)

    def f_b(index, _x, _y):
        a, b, c = index
        return b/(_x + _y)

    nd_exprs_b = np.zeros((1, 7, 1), dtype=object, order=order)
    for index in np.ndindex(*nd_exprs_b.shape):
        nd_exprs_b[index] = f_b(index, x, y)
    return args, nd_exprs_a, nd_exprs_b, f_a, f_b

@unittest.skipUnless(have_numpy, "Numpy not installed")
def test_Lambdify_Ndimensional_order_C():
    args, nd_exprs_a, nd_exprs_b, f_a, f_b = _get_Ndim_args_exprs_funcs(order='C')
    lmb4 = se.Lambdify(args, nd_exprs_a, nd_exprs_b, order='C')
    nargs = len(args)

    inp_extra_shape = (3, 5, 4)
    inp_shape = inp_extra_shape + (nargs,)
    inp4 = np.arange(reduce(mul, inp_shape)*1.0).reshape(inp_shape, order='C')
    out4a, out4b = lmb4(inp4)
    assert out4a.ndim == 7
    assert out4a.shape == inp_extra_shape + nd_exprs_a.shape
    assert out4b.ndim == 6
    assert out4b.shape == inp_extra_shape + nd_exprs_b.shape
    raises(ValueError, lambda: (lmb4(inp4.T)))
    for b, c, d in np.ndindex(inp_extra_shape):
        _x, _y = inp4[b, c, d, :]
        for index in np.ndindex(*nd_exprs_a.shape):
            assert np.isclose(out4a[(b, c, d) + index], f_a(index, _x, _y))
        for index in np.ndindex(*nd_exprs_b.shape):
            assert np.isclose(out4b[(b, c, d) + index], f_b(index, _x, _y))


@unittest.skipUnless(have_numpy, "Numpy not installed")
def test_Lambdify_Ndimensional_order_F():
    args, nd_exprs_a, nd_exprs_b, f_a, f_b = _get_Ndim_args_exprs_funcs(order='F')
    lmb4 = se.Lambdify(args, nd_exprs_a, nd_exprs_b, order='F')
    nargs = len(args)

    inp_extra_shape = (3, 5, 4)
    inp_shape = (nargs,)+inp_extra_shape
    inp4 = np.arange(reduce(mul, inp_shape)*1.0).reshape(inp_shape, order='F')
    out4a, out4b = lmb4(inp4)
    assert out4a.ndim == 7
    assert out4a.shape == nd_exprs_a.shape + inp_extra_shape
    assert out4b.ndim == 6
    assert out4b.shape == nd_exprs_b.shape + inp_extra_shape
    raises(ValueError, lambda: (lmb4(inp4.T)))
    for b, c, d in np.ndindex(inp_extra_shape):
        _x, _y = inp4[:, b, c, d]
        for index in np.ndindex(*nd_exprs_a.shape):
            assert np.isclose(out4a[index + (b, c, d)], f_a(index, _x, _y))
        for index in np.ndindex(*nd_exprs_b.shape):
            assert np.isclose(out4b[index + (b, c, d)], f_b(index, _x, _y))


@unittest.skipUnless(have_numpy, "Numpy not installed")
def test_Lambdify_inp_exceptions():
    args = x, y = se.symbols('x y')
    lmb1 = se.Lambdify([x], x**2)
    raises(ValueError, lambda: (lmb1([])))
    assert lmb1(4) == 16
    assert np.all(lmb1([4, 2]) == [16, 4])

    lmb2 = se.Lambdify(args, x**2+y**2)
    assert lmb2([2, 3]) == 13
    raises(ValueError, lambda: lmb2([]))
    raises(ValueError, lambda: lmb2([2]))
    raises(ValueError, lambda: lmb2([2, 3, 4]))
    assert np.all(lmb2([2, 3, 4, 5]) == [13, 16+25])

    def _mtx(_x, _y):
        return [
            [_x-_y, _y**2],
            [_x+_y, _x**2],
            [_x*_y, _x**_y]
        ]

    mtx = np.array(_mtx(x, y), order='F')
    lmb3 = se.Lambdify(args, mtx, order='F')
    inp3a = [2, 3]
    assert np.all(lmb3(inp3a) == _mtx(*inp3a))
    inp3b = np.array([2, 3, 4, 5, 3, 2, 1, 5])
    for inp in [inp3b, inp3b.tolist(), inp3b.reshape((2, 4), order='F')]:
        out3b = lmb3(inp)
        assert out3b.shape == (3, 2, 4)
        for i in range(4):
            assert np.all(out3b[..., i] == _mtx(*inp3b[2*i:2*(i+1)]))
    raises(ValueError, lambda: lmb3(inp3b.reshape((4, 2))))
    raises(ValueError, lambda: lmb3(inp3b.reshape((2, 4)).T))


@unittest.skipUnless(have_scipy, "Scipy not installed")
def test_scipy():
    from scipy import integrate
    import numpy as np
    args = t, x = se.symbols('t, x')
    lmb = se.Lambdify(args, [se.exp(-x*t)/t**5], as_scipy=True)
    res = integrate.nquad(lmb, [[1, np.inf], [0, np.inf]])
    assert abs(res[0] - 0.2) < 1e-7


@unittest.skipUnless(have_numpy, "Numpy not installed")
def test_as_ctypes():
    import numpy as np
    import ctypes
    x, y, z = se.symbols('x, y, z')
    l = se.Lambdify([x, y, z], [x+y+z, x*y*z+1])
    addr1, addr2 = l.as_ctypes()
    inp = np.array([1,2,3], dtype=np.double)
    out = np.array([0, 0], dtype=np.double)
    addr1(out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), inp.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), addr2)
    assert np.all(out == [6, 7])

@unittest.skipUnless(have_numpy, "Numpy not installed")
@unittest.skipUnless(se.have_llvm, "No LLVM support")
def test_llvm_float():
    import numpy as np
    import ctypes
    from symengine.lib.symengine_wrapper import LLVMFloat
    x, y, z = se.symbols('x, y, z')
    l = se.Lambdify([x, y, z], [se.Min(x, y), se.Max(y, z)], dtype=np.float32, backend='llvm')
    inp = np.array([1,2,3], dtype=np.float32)
    exp_out = np.array([1, 3], dtype=np.float32)
    out = l(inp)
    assert type(l) == LLVMFloat
    assert out.dtype == np.float32
    assert np.allclose(out, exp_out)

@unittest.skipUnless(have_numpy, "Numpy not installed")
@unittest.skipUnless(se.have_llvm, "No LLVM support")
@unittest.skipUnless(se.have_llvm_long_double, "No LLVM IEEE-80 bit support")
def test_llvm_long_double():
    import numpy as np
    import ctypes
    from symengine.lib.symengine_wrapper import LLVMLongDouble
    x, y, z = se.symbols('x, y, z')
    l = se.Lambdify([x, y, z], [2*x, y/z], dtype=np.longdouble, backend='llvm')
    inp = np.array([1,2,3], dtype=np.longdouble)
    exp_out = np.array([2, 2.0/3.0], dtype=np.longdouble)
    out = l(inp)
    assert type(l) == LLVMLongDouble
    assert out.dtype == np.longdouble
    assert np.allclose(out, exp_out)
