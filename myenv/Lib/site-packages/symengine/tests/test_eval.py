from symengine.test_utilities import raises
from symengine import (Symbol, sin, cos, Integer, Add, I, RealDouble, ComplexDouble, sqrt)

from unittest.case import SkipTest

def test_eval_double1():
    x = Symbol("x")
    y = Symbol("y")
    e = sin(x)**2 + cos(x)**2
    e = e.subs(x, 7)
    assert abs(e.n(real=True) - 1) < 1e-9
    assert abs(e.n() - 1) < 1e-9


def test_eval_double2():
    x = Symbol("x")
    e = sin(x)**2 + sqrt(2)
    raises(RuntimeError, lambda: e.n(real=True))
    assert abs(e.n() - sin(x)**2.0 - 1.414) < 1e-3

def test_n():
    x = Symbol("x")
    raises(RuntimeError, lambda: x.n(real=True))
    assert x.n() == x + 0.0

    x = 2 + I
    raises(RuntimeError, lambda: (x.n(real=True)))

    x = sqrt(Integer(4))
    y = RealDouble(2.0)
    assert x.n(real=True) == y

    x = 1 + 2*I
    y = 1.0 + 2.0*I
    assert x.n() == y


def test_n_mpfr():
    x = sqrt(Integer(2))
    try:
        from symengine import RealMPFR
        y = RealMPFR('1.41421356237309504880169', 75)
        assert x.n(75, real=True) == y
    except ImportError:
        raises(ValueError, lambda: (x.n(75, real=True)))
        raises(ValueError, lambda: (x.n(75)))
        raise SkipTest("No MPFR support")


def test_n_mpc():
    x = sqrt(Integer(2)) + 3*I
    try:
        from symengine import ComplexMPC
        y = ComplexMPC('1.41421356237309504880169', '3.0', 75)
        assert x.n(75) == y
    except ImportError:
        raises(Exception, lambda: (x.n(75, real=True)))
        raises(ValueError, lambda: (x.n(75, real=False)))
        raises(ValueError, lambda: (x.n(75)))
        raise SkipTest("No MPC support")


def test_rel():
    x = Symbol("x")
    y = Symbol("y")
    ex = (x + y < x)
    assert repr(ex) == "x + y < x"
