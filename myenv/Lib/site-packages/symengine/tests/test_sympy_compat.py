from symengine.sympy_compat import (Integer, Rational, S, Basic, Add, Mul,
    Pow, symbols, Symbol, log, sin, cos, sech, csch, zeros, atan2, nan, Number, Float,
    Min, Max, RealDouble, have_mpfr, Abs)
from symengine.test_utilities import raises


def test_Integer():
    i = Integer(5)
    assert isinstance(i, Integer)
    assert isinstance(i, Rational)
    assert isinstance(i, Number)
    assert isinstance(i, Basic)
    assert i.p == 5
    assert i.q == 1


def test_Rational():
    i = S(1)/2
    assert isinstance(i, Rational)
    assert isinstance(i, Number)
    assert isinstance(i, Basic)
    assert i.p == 1
    assert i.q == 2
    x = symbols("x")
    assert not isinstance(x, Rational)
    assert not isinstance(x, Number)


def test_Float():
    A = Float("1.23", precision = 53)
    B = Float("1.23")
    C = Float(A)
    assert A == B == C
    assert isinstance(A, Float)
    assert isinstance(B, Float)
    assert isinstance(C, Float)
    assert isinstance(A, RealDouble)
    assert isinstance(B, RealDouble)
    assert isinstance(C, RealDouble)
    raises(ValueError, lambda: Float("1.23", dps = 3, precision = 10))
    raises(ValueError, lambda: Float(A, dps = 3, precision = 16))
    if have_mpfr:
        from symengine.sympy_compat import RealMPFR
        A = Float("1.23", dps = 16)
        B = Float("1.23", precision = 56)
        assert A == B
        assert isinstance(A, Float)
        assert isinstance(B, Float)
        assert isinstance(A, RealMPFR)
        assert isinstance(B, RealMPFR)
        A = Float(C, dps = 16)
        assert A == B
        assert isinstance(A, Float)
        assert isinstance(A, RealMPFR)
        A = Float(A, precision = 53)
        assert A == C
        assert isinstance(A, Float)
        assert isinstance(A, RealDouble)
    if not have_mpfr:
        raises(ValueError, lambda: Float("1.23", precision = 58))        


def test_Add():
    x, y = symbols("x y")
    i = Add(x, x)
    assert isinstance(i, Mul)
    i = Add(x, y)
    assert isinstance(i, Add)
    assert isinstance(i, Basic)
    assert i.func(y, x) == i


def test_Mul():
    x, y = symbols("x y")
    i = Mul(x, x)
    assert isinstance(i, Pow)
    i = Mul(x, y)
    assert isinstance(i, Mul)
    assert isinstance(i, Basic)
    assert i.func(y, x) == i


def test_Pow():
    x = symbols("x")
    i = Pow(x, 1)
    assert isinstance(i, Symbol)
    i = Pow(x, 2)
    assert isinstance(i, Pow)
    assert isinstance(i, Basic)
    assert i.func(x, 2) == i


def test_Max():
    x = Symbol("x")
    y = Symbol("y")
    z = Symbol("z")
    assert Max(Integer(6)/3, 1) == 2
    assert Max(-2, 2) == 2
    assert Max(2, 2) == 2
    assert Max(0.2, 0.3) == 0.3
    assert Max(x, x) == x
    assert Max(x, y) == Max(y, x)
    assert Max(x, y, z) == Max(z, y, x)
    assert Max(x, Max(y, z)) == Max(z, y, x)
    assert Max(1000, 100, -100, x, y, z) == Max(x, y, z, 1000)
    assert Max(cos(x), sin(x)) == Max(sin(x), cos(x))


def test_Min():
    x = Symbol("x")
    y = Symbol("y")
    z = Symbol("z")
    assert Min(Integer(6)/3, 1) == 1
    assert Min(-2, 2) == -2
    assert Min(2, 2) == 2
    assert Min(0.2, 0.3) == 0.2
    assert Min(x, x) == x
    assert Min(x, y) == Min(y, x)
    assert Min(x, y, z) == Min(z, y, x)
    assert Min(x, Min(y, z)) == Min(z, y, x)
    assert Min(1000, 100, -100, x, y, z) == Min(x, y, z, -100)
    assert Min(cos(x), sin(x)) == Min(cos(x), sin(x))


def test_NaN():
    type(nan)() == nan


def test_sin():
    x = symbols("x")
    i = sin(0)
    assert isinstance(i, Integer)
    i = sin(x)
    assert isinstance(i, sin)


def test_sech():
    x = symbols("x")
    i = sech(0)
    assert isinstance(i, Integer)
    i = sech(x)
    assert isinstance(i, sech)


def test_csch():
    x = symbols("x")
    i = csch(x)
    assert isinstance(i, csch)
    i = csch(-1)
    j = csch(1)
    assert i == -j


def test_log():
    x, y = symbols("x y")
    i = log(x, y)
    assert isinstance(i, Mul)
    i = log(x)
    assert isinstance(i, log)


def test_ATan2():
    x, y = symbols("x y")
    i = atan2(x, y)
    assert isinstance(i, atan2)
    i = atan2(0, 1)
    assert i == 0


def test_zeros():
    assert zeros(3, c=2).shape == (3, 2)


def test_has_functions_module():
    import symengine.sympy_compat as sp
    assert sp.functions.sin(0) == 0


def test_subclass_symbol():
    # Subclass of Symbol with an extra attribute
    class Wrapper(Symbol):
        def __new__(cls, name, extra_attribute):
            return Symbol.__new__(cls, name)

        def __init__(self, name, extra_attribute):
            super().__init__(name)
            self.extra_attribute = extra_attribute

    # Instantiate the subclass
    x = Wrapper("x", extra_attribute=3)
    assert x.extra_attribute == 3
    two_x = 2 * x
    # Check that after arithmetic, same subclass is returned
    assert two_x.args[1] is x
    del two_x
    x._unsafe_reset()

def test_Abs():
    x = symbols("x")
    assert Abs(x) == Abs(-x)
