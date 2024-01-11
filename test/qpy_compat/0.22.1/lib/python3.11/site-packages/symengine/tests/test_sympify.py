from symengine.test_utilities import raises

from symengine import (Symbol, Integer, sympify, SympifyError, true, false, pi, nan, oo,
                       zoo, E, I, GoldenRatio, Catalan, Rational, sqrt, Eq)
from symengine.lib.symengine_wrapper import _sympify, S, One, polygamma


def test_sympify1():
    assert sympify(1) == Integer(1)
    assert sympify(2) != Integer(1)
    assert sympify(-5) == Integer(-5)
    assert sympify(Integer(3)) == Integer(3)
    assert sympify(('0', '0')) == (0, 0)
    assert sympify(['0', '0']) == [0, 0]
    assert sympify("3+5") == Integer(8)
    assert true == sympify(True)
    assert false == sympify(False)


def test_S():
    assert S(0) == Integer(0)
    assert S(1) == Integer(1)
    assert S(-1) == Integer(-1)
    assert S(1) / 2 == Rational(1, 2)
    assert S.One is S(1)
    assert S.Zero is S(0)
    assert S.NegativeOne is S(-1)
    assert S.Half is S(1) / 2
    assert S.Pi is pi
    assert S.NaN is S(0) / 0
    assert S.Infinity is -oo * -10
    assert S.NegativeInfinity is oo * (-3)
    assert S.ComplexInfinity is zoo
    assert S.Exp1 is (E + 1 - 1)
    assert S.ImaginaryUnit is sqrt(-1)
    assert S.GoldenRatio * 2 / 2 is GoldenRatio
    assert S.Catalan * 1 is Catalan
    assert S.EulerGamma is polygamma(0, 1) * -1
    assert S.true is Eq(2, 2)
    assert S.false is Eq(2, 3)
    assert S(1) / 0 is zoo
    assert S.Pi * 1 is pi
    assert type(S.One) == One


def test_sympify_error1a():
    class Test:
        pass
    raises(SympifyError, lambda: sympify(Test()))


def test_sympify_error1b():
    assert not _sympify("1***2", raise_error=False)


def test_error1():
    # _sympify doesn't parse strings
    raises(SympifyError, lambda: _sympify("x"))


def test_sympify_pow():
    # https://github.com/symengine/symengine.py/issues/251
    assert sympify('y*pow(x, -1)') == Symbol('y')/Symbol('x')
