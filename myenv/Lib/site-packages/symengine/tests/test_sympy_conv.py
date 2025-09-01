from symengine import (Symbol, Integer, sympify, SympifyError, log,
        function_symbol, I, E, pi, oo, zoo, nan, true, false,
        exp, gamma, have_mpfr, have_mpc, DenseMatrix, sin, cos, tan, cot,
        csc, sec, asin, acos, atan, acot, acsc, asec, sinh, cosh, tanh, coth,
        asinh, acosh, atanh, acoth, atan2, Add, Mul, Pow, diff, GoldenRatio,
        Catalan, EulerGamma, UnevaluatedExpr, RealDouble)
from symengine.lib.symengine_wrapper import (Subs, Derivative, RealMPFR,
        ComplexMPC, PyNumber, Function, LambertW, zeta, dirichlet_eta,
        KroneckerDelta, LeviCivita, erf, erfc, lowergamma, uppergamma,
        loggamma, beta, polygamma, sign, floor, ceiling, conjugate, And,
        Or, Not, Xor, Piecewise, Interval, EmptySet, FiniteSet, Contains,
        Union, Complement, UniversalSet, Reals, Rationals, Integers)
import unittest

# Note: We test _sympy_() for SymEngine -> SymPy conversion, as those are
# methods that are implemented in this library. Users can simply use
# sympy.sympify(...). To do this conversion, as this function will call
# our _sympy_() methods under the hood.
#
# For SymPy -> SymEngine, we test symengine.sympify(...) which
# does the conversion.

try:
    import sympy
    from sympy.core.cache import clear_cache
    import atexit
    atexit.register(clear_cache)
    have_sympy = True
except ImportError:
    have_sympy = False


@unittest.skipIf(not have_sympy, "SymPy not installed")
def test_conv1():
    x = Symbol("x")
    assert x._sympy_() == sympy.Symbol("x")
    assert x._sympy_() != sympy.Symbol("y")
    x = Symbol("y")
    assert x._sympy_() != sympy.Symbol("x")
    assert x._sympy_() == sympy.Symbol("y")


@unittest.skipIf(not have_sympy, "SymPy not installed")
def test_conv1b():
    x = sympy.Symbol("x")
    assert sympify(x) == Symbol("x")
    assert sympify(x) != Symbol("y")
    x = sympy.Symbol("y")
    assert sympify(x) != Symbol("x")
    assert sympify(x) == Symbol("y")


@unittest.skipIf(not have_sympy, "SymPy not installed")
def test_conv2():
    x = Symbol("x")
    y = Symbol("y")
    z = Symbol("z")
    e = x*y
    assert e._sympy_() == sympy.Symbol("x")*sympy.Symbol("y")
    e = x*y*z
    assert e._sympy_() == sympy.Symbol("x")*sympy.Symbol("y")*sympy.Symbol("z")


@unittest.skipIf(not have_sympy, "SymPy not installed")
def test_conv2b():
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    z = sympy.Symbol("z")
    e = x*y
    assert sympify(e) == Symbol("x")*Symbol("y")
    e = x*y*z
    assert sympify(e) == Symbol("x")*Symbol("y")*Symbol("z")


@unittest.skipIf(not have_sympy, "SymPy not installed")
def test_conv3():
    x = Symbol("x")
    y = Symbol("y")
    z = Symbol("z")
    e = x+y
    assert e._sympy_() == sympy.Symbol("x")+sympy.Symbol("y")
    e = x+y+z
    assert e._sympy_() == sympy.Symbol("x")+sympy.Symbol("y")+sympy.Symbol("z")


@unittest.skipIf(not have_sympy, "SymPy not installed")
def test_conv3b():
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    z = sympy.Symbol("z")
    e = x+y
    assert sympify(e) == Symbol("x")+Symbol("y")
    e = x+y+z
    assert sympify(e) == Symbol("x")+Symbol("y")+Symbol("z")


@unittest.skipIf(not have_sympy, "SymPy not installed")
def test_conv4():
    x = Symbol("x")
    y = Symbol("y")
    z = Symbol("z")
    e = x**y
    assert e._sympy_() == sympy.Symbol("x")**sympy.Symbol("y")
    e = (x+y)**z
    assert (e._sympy_() ==
            (sympy.Symbol("x")+sympy.Symbol("y"))**sympy.Symbol("z"))


@unittest.skipIf(not have_sympy, "SymPy not installed")
def test_conv4b():
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    z = sympy.Symbol("z")
    e = x**y
    assert sympify(e) == Symbol("x")**Symbol("y")
    e = (x+y)**z
    assert sympify(e) == (Symbol("x")+Symbol("y"))**Symbol("z")


@unittest.skipIf(not have_sympy, "SymPy not installed")
def test_conv5():
    x = Integer(5)
    y = Integer(6)
    assert x._sympy_() == sympy.Integer(5)
    assert (x/y)._sympy_() == sympy.Integer(5) / sympy.Integer(6)


@unittest.skipIf(not have_sympy, "SymPy not installed")
def test_conv5b():
    x = sympy.Integer(5)
    y = sympy.Integer(6)
    assert sympify(x) == Integer(5)
    assert sympify(x/y) == Integer(5) / Integer(6)


@unittest.skipIf(not have_sympy, "SymPy not installed")
def test_conv6():
    x = Symbol("x")
    y = Symbol("y")
    assert (x/3)._sympy_() == sympy.Symbol("x") / 3
    assert (3*x)._sympy_() == 3*sympy.Symbol("x")
    assert (3+x)._sympy_() == 3+sympy.Symbol("x")
    assert (3-x)._sympy_() == 3-sympy.Symbol("x")
    assert (x/y)._sympy_() == sympy.Symbol("x") / sympy.Symbol("y")


@unittest.skipIf(not have_sympy, "SymPy not installed")
def test_conv6b():
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    assert sympify(x/3) == Symbol("x") / 3
    assert sympify(3*x) == 3*Symbol("x")
    assert sympify(3+x) == 3+Symbol("x")
    assert sympify(3-x) == 3-Symbol("x")
    assert sympify(x/y) == Symbol("x") / Symbol("y")


@unittest.skipIf(not have_sympy, "SymPy not installed")
def test_conv7():
    x = Symbol("x")
    y = Symbol("y")
    assert sin(x/3) == sin(sympy.Symbol("x") / 3)
    assert cos(x/3) == cos(sympy.Symbol("x") / 3)
    assert tan(x/3) == tan(sympy.Symbol("x") / 3)
    assert cot(x/3) == cot(sympy.Symbol("x") / 3)
    assert csc(x/3) == csc(sympy.Symbol("x") / 3)
    assert sec(x/3) == sec(sympy.Symbol("x") / 3)
    assert asin(x/3) == asin(sympy.Symbol("x") / 3)
    assert acos(x/3) == acos(sympy.Symbol("x") / 3)
    assert atan(x/3) == atan(sympy.Symbol("x") / 3)
    assert acot(x/3) == acot(sympy.Symbol("x") / 3)
    assert acsc(x/3) == acsc(sympy.Symbol("x") / 3)
    assert asec(x/3) == asec(sympy.Symbol("x") / 3)
    assert atan2(x/3, y) == atan2(sympy.Symbol("x") / 3, sympy.Symbol("y"))

    assert sin(x/3)._sympy_() == sympy.sin(sympy.Symbol("x") / 3)
    assert sin(x/3)._sympy_() != sympy.cos(sympy.Symbol("x") / 3)
    assert cos(x/3)._sympy_() == sympy.cos(sympy.Symbol("x") / 3)
    assert tan(x/3)._sympy_() == sympy.tan(sympy.Symbol("x") / 3)
    assert cot(x/3)._sympy_() == sympy.cot(sympy.Symbol("x") / 3)
    assert csc(x/3)._sympy_() == sympy.csc(sympy.Symbol("x") / 3)
    assert sec(x/3)._sympy_() == sympy.sec(sympy.Symbol("x") / 3)
    assert asin(x/3)._sympy_() == sympy.asin(sympy.Symbol("x") / 3)
    assert acos(x/3)._sympy_() == sympy.acos(sympy.Symbol("x") / 3)
    assert atan(x/3)._sympy_() == sympy.atan(sympy.Symbol("x") / 3)
    assert acot(x/3)._sympy_() == sympy.acot(sympy.Symbol("x") / 3)
    assert acsc(x/3)._sympy_() == sympy.acsc(sympy.Symbol("x") / 3)
    assert asec(x/3)._sympy_() == sympy.asec(sympy.Symbol("x") / 3)
    assert atan2(x/3, y)._sympy_() == sympy.atan2(sympy.Symbol("x") / 3, sympy.Symbol("y"))

    assert sympy.sympify(sin(x/3)) == sympy.sin(sympy.Symbol("x") / 3)
    assert sympy.sympify(sin(x/3)) != sympy.cos(sympy.Symbol("x") / 3)
    assert sympy.sympify(cos(x/3)) == sympy.cos(sympy.Symbol("x") / 3)
    assert sympy.sympify(tan(x/3)) == sympy.tan(sympy.Symbol("x") / 3)
    assert sympy.sympify(cot(x/3)) == sympy.cot(sympy.Symbol("x") / 3)
    assert sympy.sympify(csc(x/3)) == sympy.csc(sympy.Symbol("x") / 3)
    assert sympy.sympify(sec(x/3)) == sympy.sec(sympy.Symbol("x") / 3)
    assert sympy.sympify(asin(x/3)) == sympy.asin(sympy.Symbol("x") / 3)
    assert sympy.sympify(acos(x/3)) == sympy.acos(sympy.Symbol("x") / 3)
    assert sympy.sympify(atan(x/3)) == sympy.atan(sympy.Symbol("x") / 3)
    assert sympy.sympify(acot(x/3)) == sympy.acot(sympy.Symbol("x") / 3)
    assert sympy.sympify(acsc(x/3)) == sympy.acsc(sympy.Symbol("x") / 3)
    assert sympy.sympify(asec(x/3)) == sympy.asec(sympy.Symbol("x") / 3)
    assert sympy.sympify(atan2(x/3, y)) == sympy.atan2(sympy.Symbol("x") / 3, sympy.Symbol("y"))


@unittest.skipIf(not have_sympy, "SymPy not installed")
def test_conv7b():
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    assert sympify(sympy.sin(x/3)) == sin(Symbol("x") / 3)
    assert sympify(sympy.sin(x/3)) != cos(Symbol("x") / 3)
    assert sympify(sympy.cos(x/3)) == cos(Symbol("x") / 3)
    assert sympify(sympy.tan(x/3)) == tan(Symbol("x") / 3)
    assert sympify(sympy.cot(x/3)) == cot(Symbol("x") / 3)
    assert sympify(sympy.csc(x/3)) == csc(Symbol("x") / 3)
    assert sympify(sympy.sec(x/3)) == sec(Symbol("x") / 3)
    assert sympify(sympy.asin(x/3)) == asin(Symbol("x") / 3)
    assert sympify(sympy.acos(x/3)) == acos(Symbol("x") / 3)
    assert sympify(sympy.atan(x/3)) == atan(Symbol("x") / 3)
    assert sympify(sympy.acot(x/3)) == acot(Symbol("x") / 3)
    assert sympify(sympy.acsc(x/3)) == acsc(Symbol("x") / 3)
    assert sympify(sympy.asec(x/3)) == asec(Symbol("x") / 3)
    assert sympify(sympy.atan2(x/3, y)) == atan2(Symbol("x") / 3, Symbol("y"))


@unittest.skipIf(not have_sympy, "SymPy not installed")
def test_conv8():
    e1 = function_symbol("f", Symbol("x"))
    e2 = function_symbol("g", Symbol("x"), Symbol("y"))
    assert e1._sympy_() == sympy.Function("f")(sympy.Symbol("x"))
    assert e2._sympy_() != sympy.Function("f")(sympy.Symbol("x"))
    assert (e2._sympy_() ==
            sympy.Function("g")(sympy.Symbol("x"), sympy.Symbol("y")))

    e3 = function_symbol("q", Symbol("t"))
    assert e3._sympy_() == sympy.Function("q")(sympy.Symbol("t"))
    assert e3._sympy_() != sympy.Function("f")(sympy.Symbol("t"))
    assert (e3._sympy_() !=
            sympy.Function("q")(sympy.Symbol("t"), sympy.Symbol("t")))


@unittest.skipIf(not have_sympy, "SymPy not installed")
def test_conv8b():
    e1 = sympy.Function("f")(sympy.Symbol("x"))
    e2 = sympy.Function("g")(sympy.Symbol("x"), sympy.Symbol("y"))
    assert sympify(e1) == function_symbol("f", Symbol("x"))
    assert sympify(e2) != function_symbol("f", Symbol("x"))
    assert sympify(e2) == function_symbol("g", Symbol("x"), Symbol("y"))

    e3 = sympy.Function("q")(sympy.Symbol("t"))
    assert sympify(e3) == function_symbol("q", Symbol("t"))
    assert sympify(e3) != function_symbol("f", Symbol("t"))
    assert sympify(e3) != function_symbol("q", Symbol("t"), Symbol("t"))


@unittest.skipIf(not have_sympy, "SymPy not installed")
def test_conv9():
    x = Symbol("x")
    y = Symbol("y")
    assert (I)._sympy_() == sympy.I
    assert (2*I+3)._sympy_() == 2*sympy.I+3
    assert (2*I/5+Integer(3)/5)._sympy_() == 2*sympy.I/5+sympy.S(3)/5
    assert (x*I+3)._sympy_() == sympy.Symbol("x")*sympy.I + 3
    assert (x+I*y)._sympy_() == sympy.Symbol("x") + sympy.I*sympy.Symbol("y")


@unittest.skipIf(not have_sympy, "SymPy not installed")
def test_conv9b():
    x = Symbol("x")
    y = Symbol("y")
    assert sympify(sympy.I) == I
    assert sympify(2*sympy.I+3) == 2*I+3
    assert sympify(2*sympy.I/5+sympy.S(3)/5) == 2*I/5+Integer(3)/5
    assert sympify(sympy.Symbol("x")*sympy.I + 3) == x*I+3
    assert sympify(sympy.Symbol("x") + sympy.I*sympy.Symbol("y")) == x+I*y


@unittest.skipIf(not have_sympy, "SymPy not installed")
def test_conv10():
    A = DenseMatrix(1, 4, [Integer(1), Integer(2), Integer(3), Integer(4)])
    assert (A._sympy_() == sympy.Matrix(1, 4,
                                        [sympy.Integer(1), sympy.Integer(2),
                                         sympy.Integer(3), sympy.Integer(4)]))

    B = DenseMatrix(4, 1, [Symbol("x"), Symbol("y"), Symbol("z"), Symbol("t")])
    assert (B._sympy_() == sympy.Matrix(4, 1,
                                        [sympy.Symbol("x"), sympy.Symbol("y"),
                                         sympy.Symbol("z"), sympy.Symbol("t")])
            )

    C = DenseMatrix(2, 2,
                    [Integer(5), Symbol("x"),
                     function_symbol("f", Symbol("x")), 1 + I])

    assert (C._sympy_() ==
            sympy.Matrix([[5, sympy.Symbol("x")],
                          [sympy.Function("f")(sympy.Symbol("x")),
                           1 + sympy.I]]))


@unittest.skipIf(not have_sympy, "SymPy not installed")
def test_conv10b():
    A = sympy.Matrix([[sympy.Symbol("x"), sympy.Symbol("y")],
                     [sympy.Symbol("z"), sympy.Symbol("t")]])
    assert sympify(A) == DenseMatrix(2, 2, [Symbol("x"), Symbol("y"),
                                            Symbol("z"), Symbol("t")])

    B = sympy.Matrix([[1, 2], [3, 4]])
    assert sympify(B) == DenseMatrix(2, 2, [Integer(1), Integer(2), Integer(3),
                                            Integer(4)])

    C = sympy.Matrix([[7, sympy.Symbol("y")],
                     [sympy.Function("g")(sympy.Symbol("z")), 3 + 2*sympy.I]])
    assert sympify(C) == DenseMatrix(2, 2, [Integer(7), Symbol("y"),
                                            function_symbol("g",
                                                            Symbol("z")),
                                            3 + 2*I])


@unittest.skipIf(not have_sympy, "SymPy not installed")
def test_conv11():
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    x1 = Symbol("x")
    y1 = Symbol("y")
    f = sympy.Function("f")
    f1 = Function("f")

    e1 = diff(f(2*x, y), x)
    e2 = diff(f1(2*x1, y1), x1)
    e3 = diff(f1(2*x1, y1), y1)

    assert sympify(e1) == e2
    assert sympify(e1) != e3

    assert e2._sympy_() == e1
    assert e3._sympy_() != e1


@unittest.skipIf(not have_sympy, "SymPy not installed")
def test_conv12():
    x = Symbol("x")
    y = Symbol("y")
    assert sinh(x/3) == sinh(sympy.Symbol("x") / 3)
    assert cosh(x/3) == cosh(sympy.Symbol("x") / 3)
    assert tanh(x/3) == tanh(sympy.Symbol("x") / 3)
    assert coth(x/3) == coth(sympy.Symbol("x") / 3)
    assert asinh(x/3) == asinh(sympy.Symbol("x") / 3)
    assert acosh(x/3) == acosh(sympy.Symbol("x") / 3)
    assert atanh(x/3) == atanh(sympy.Symbol("x") / 3)
    assert acoth(x/3) == acoth(sympy.Symbol("x") / 3)

    assert sinh(x/3)._sympy_() == sympy.sinh(sympy.Symbol("x") / 3)
    assert cosh(x/3)._sympy_() == sympy.cosh(sympy.Symbol("x") / 3)
    assert tanh(x/3)._sympy_() == sympy.tanh(sympy.Symbol("x") / 3)
    assert coth(x/3)._sympy_() == sympy.coth(sympy.Symbol("x") / 3)
    assert asinh(x/3)._sympy_() == sympy.asinh(sympy.Symbol("x") / 3)
    assert acosh(x/3)._sympy_() == sympy.acosh(sympy.Symbol("x") / 3)
    assert atanh(x/3)._sympy_() == sympy.atanh(sympy.Symbol("x") / 3)
    assert acoth(x/3)._sympy_() == sympy.acoth(sympy.Symbol("x") / 3)


@unittest.skipIf(not have_sympy, "SymPy not installed")
def test_conv12b():
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    assert sympify(sympy.sinh(x/3)) == sinh(Symbol("x") / 3)
    assert sympify(sympy.cosh(x/3)) == cosh(Symbol("x") / 3)
    assert sympify(sympy.tanh(x/3)) == tanh(Symbol("x") / 3)
    assert sympify(sympy.coth(x/3)) == coth(Symbol("x") / 3)
    assert sympify(sympy.asinh(x/3)) == asinh(Symbol("x") / 3)
    assert sympify(sympy.acosh(x/3)) == acosh(Symbol("x") / 3)
    assert sympify(sympy.atanh(x/3)) == atanh(Symbol("x") / 3)
    assert sympify(sympy.acoth(x/3)) == acoth(Symbol("x") / 3)


@unittest.skipIf(not have_sympy, "SymPy not installed")
def test_tuples_lists():
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    z = sympy.Symbol("z")
    L = [x, y, z, x*y, z**y]
    t = (x, y, z, x*y, z**y)
    x = Symbol("x")
    y = Symbol("y")
    z = Symbol("z")
    l2 = [x, y, z, x*y, z**y]
    t2 = (x, y, z, x*y, z**y)
    assert sympify(L) == l2
    assert sympify(t) == t2
    assert sympify(L) != t2
    assert sympify(t) != l2

    assert L == l2
    assert t == t2
    assert L != t2
    assert t != l2


@unittest.skipIf(not have_sympy, "SymPy not installed")
def test_exp():
    x = Symbol("x")
    e1 = sympy.exp(sympy.Symbol("x"))
    e2 = exp(x)
    assert sympify(e1) == e2
    assert e1 == e2._sympy_()

    e1 = sympy.exp(sympy.Symbol("x")).diff(sympy.Symbol("x"))
    e2 = exp(x).diff(x)
    assert sympify(e1) == e2
    assert e1 == e2._sympy_()


@unittest.skipIf(not have_sympy, "SymPy not installed")
def test_gamma():
    x = Symbol("x")
    e1 = sympy.gamma(sympy.Symbol("x"))
    e2 = gamma(x)
    assert sympify(e1) == e2
    assert e1 == e2._sympy_()


@unittest.skipIf(not have_sympy, "SymPy not installed")
def test_constants():
    assert sympify(sympy.E) == E
    assert sympy.E == E._sympy_()

    assert sympify(sympy.pi) == pi
    assert sympy.pi == pi._sympy_()

    assert sympify(sympy.GoldenRatio) == GoldenRatio
    assert sympy.GoldenRatio == GoldenRatio._sympy_()

    assert sympify(sympy.Catalan) == Catalan
    assert sympy.Catalan == Catalan._sympy_()

    assert sympify(sympy.EulerGamma) == EulerGamma
    assert sympy.EulerGamma == EulerGamma._sympy_()

    assert sympify(sympy.oo) == oo
    assert sympy.oo == oo._sympy_()

    assert sympify(sympy.zoo) == zoo
    assert sympy.zoo == zoo._sympy_()

    assert sympify(sympy.nan) == nan
    assert sympy.nan == nan._sympy_()


@unittest.skipIf(not have_sympy, "SymPy not installed")
def test_booleans():
    assert sympify(sympy.S.true) == true
    assert sympy.S.true == true._sympy_()

    assert sympify(sympy.S.false) == false
    assert sympy.S.false == false._sympy_()


@unittest.skipIf(not have_sympy, "SymPy not installed")
def test_abs():
    x = Symbol("x")
    e1 = abs(sympy.Symbol("x"))
    e2 = abs(x)
    assert sympify(e1) == e2
    assert e1 == e2._sympy_()

    e1 = abs(2*sympy.Symbol("x"))
    e2 = 2*abs(x)
    assert sympify(e1) == e2
    assert e1 == e2._sympy_()

    y = Symbol("y")
    e1 = abs(sympy.Symbol("y")*sympy.Symbol("x"))
    e2 = abs(y*x)
    assert sympify(e1) == e2
    assert e1 == e2._sympy_()


@unittest.skipIf(not have_sympy, "SymPy not installed")
def test_mpfr():
    if have_mpfr:
        a = RealMPFR('100', 100)
        b = sympy.Float('100', 29)
        assert sympify(b) == a
        assert b == a._sympy_()


@unittest.skipIf(not have_sympy, "SymPy not installed")
def test_mpc():
    if have_mpc:
        a = ComplexMPC('1', '2', 100)
        b = sympy.Float(1, 29) + sympy.Float(2, 29) * sympy.I
        assert sympify(b) == a
        assert b == a._sympy_()


@unittest.skipIf(not have_sympy, "SymPy not installed")
def test_log():
    x = Symbol("x")
    x1 = sympy.Symbol("x")

    assert log(x) == log(x1)
    assert log(x)._sympy_() == sympy.log(x1)
    assert sympify(sympy.log(x1)) == log(x)

    y = Symbol("y")
    y1 = sympy.Symbol("y")

    assert log(x, y) == log(x, y1)
    assert log(x1, y) == log(x1, y1)
    assert log(x, y)._sympy_() == sympy.log(x1, y1)
    assert sympify(sympy.log(x1, y1)) == log(x, y)


@unittest.skipIf(not have_sympy, "SymPy not installed")
def test_lambertw():
    x = Symbol("x")
    e1 = sympy.LambertW(sympy.Symbol("x"))
    e2 = LambertW(x)
    assert sympify(e1) == e2
    assert e2._sympy_() == e1


@unittest.skipIf(not have_sympy, "SymPy not installed")
def test_zeta():
    x = Symbol("x")
    y = Symbol("y")
    e1 = sympy.zeta(sympy.Symbol("x"))
    e2 = zeta(x)
    assert sympify(e1) == e2
    assert e2._sympy_() == e1
    e1 = sympy.zeta(sympy.Symbol("x"), sympy.Symbol("y"))
    e2 = zeta(x, y)
    assert sympify(e1) == e2
    assert e2._sympy_() == e1


@unittest.skipIf(not have_sympy, "SymPy not installed")
def test_dirichlet_eta():
    x = Symbol("x")
    e1 = sympy.dirichlet_eta(sympy.Symbol("x"))
    e2 = dirichlet_eta(x)
    assert sympify(e1) == e2
    assert e2._sympy_() == e1


@unittest.skipIf(not have_sympy, "SymPy not installed")
def test_kronecker_delta():
    x = Symbol("x")
    y = Symbol("y")
    e1 = sympy.KroneckerDelta(sympy.Symbol("x"), sympy.Symbol("y"))
    e2 = KroneckerDelta(x, y)
    assert sympify(e1) == e2
    assert e2._sympy_() == e1


@unittest.skipIf(not have_sympy, "SymPy not installed")
def test_levi_civita():
    x = Symbol("x")
    y = Symbol("y")
    z = Symbol("z")
    e1 = sympy.LeviCivita(sympy.Symbol("x"), sympy.Symbol("y"), sympy.Symbol("z"))
    e2 = LeviCivita(x, y, z)
    assert sympify(e1) == e2
    assert e2._sympy_() == e1


@unittest.skipIf(not have_sympy, "SymPy not installed")
def test_erf():
    x = Symbol("x")
    e1 = sympy.erf(sympy.Symbol("x"))
    e2 = erf(x)
    assert sympify(e1) == e2
    assert e2._sympy_() == e1


@unittest.skipIf(not have_sympy, "SymPy not installed")
def test_erfc():
    x = Symbol("x")
    e1 = sympy.erfc(sympy.Symbol("x"))
    e2 = erfc(x)
    assert sympify(e1) == e2
    assert e2._sympy_() == e1


@unittest.skipIf(not have_sympy, "SymPy not installed")
def test_lowergamma():
    x = Symbol("x")
    y = Symbol("y")
    e1 = sympy.lowergamma(sympy.Symbol("x"), sympy.Symbol("y"))
    e2 = lowergamma(x, y)
    assert sympify(e1) == e2
    assert e2._sympy_() == e1


@unittest.skipIf(not have_sympy, "SymPy not installed")
def test_uppergamma():
    x = Symbol("x")
    y = Symbol("y")
    e1 = sympy.uppergamma(sympy.Symbol("x"), sympy.Symbol("y"))
    e2 = uppergamma(x, y)
    assert sympify(e1) == e2
    assert e2._sympy_() == e1


@unittest.skipIf(not have_sympy, "SymPy not installed")
def test_loggamma():
    x = Symbol("x")
    e1 = sympy.loggamma(sympy.Symbol("x"))
    e2 = loggamma(x)
    assert sympify(e1) == e2
    assert e2._sympy_() == e1


@unittest.skipIf(not have_sympy, "SymPy not installed")
def test_beta():
    x = Symbol("x")
    y = Symbol("y")
    e1 = sympy.beta(sympy.Symbol("y"), sympy.Symbol("x"))
    e2 = beta(y, x)
    assert sympify(e1) == e2
    assert e2._sympy_() == e1


@unittest.skipIf(not have_sympy, "SymPy not installed")
def test_polygamma():
    x = Symbol("x")
    y = Symbol("y")
    e1 = sympy.polygamma(sympy.Symbol("x"), sympy.Symbol("y"))
    e2 = polygamma(x, y)
    assert sympify(e1) == e2
    assert e2._sympy_() == e1


@unittest.skipIf(not have_sympy, "SymPy not installed")
def test_sign():
    x = Symbol("x")
    e1 = sympy.sign(sympy.Symbol("x"))
    e2 = sign(x)
    assert sympify(e1) == e2
    assert e2._sympy_() == e1


@unittest.skipIf(not have_sympy, "SymPy not installed")
def test_floor():
    x = Symbol("x")
    e1 = sympy.floor(sympy.Symbol("x"))
    e2 = floor(x)
    assert sympify(e1) == e2
    assert e2._sympy_() == e1


@unittest.skipIf(not have_sympy, "SymPy not installed")
def test_ceiling():
    x = Symbol("x")
    e1 = sympy.ceiling(sympy.Symbol("x"))
    e2 = ceiling(x)
    assert sympify(e1) == e2
    assert e2._sympy_() == e1


@unittest.skipIf(not have_sympy, "SymPy not installed")
def test_conjugate():
    x = Symbol("x")
    e1 = sympy.conjugate(sympy.Symbol("x"))
    e2 = conjugate(x)
    assert sympify(e1) == e2
    assert e2._sympy_() == e1


@unittest.skipIf(not have_sympy, "SymPy not installed")
def test_unevaluated_expr():
    x = Symbol("x")
    e1 = sympy.UnevaluatedExpr(sympy.Symbol("x"))
    e2 = UnevaluatedExpr(x)
    assert sympify(e1) == e2
    assert e2._sympy_() == e1


@unittest.skipIf(not have_sympy, "SymPy not installed")
def test_logic():
    x = true
    y = false
    x1 = sympy.true
    y1 = sympy.false

    assert And(x, y) == And(x1, y1)
    assert And(x1, y) == And(x1, y1)
    assert And(x, y)._sympy_() == sympy.And(x1, y1)
    assert sympify(sympy.And(x1, y1)) == And(x, y)

    assert Or(x, y) == Or(x1, y1)
    assert Or(x1, y) == Or(x1, y1)
    assert Or(x, y)._sympy_() == sympy.Or(x1, y1)
    assert sympify(sympy.Or(x1, y1)) == Or(x, y)

    assert Not(x) == Not(x1)
    assert Not(x1) == Not(x1)
    assert Not(x)._sympy_() == sympy.Not(x1)
    assert sympify(sympy.Not(x1)) == Not(x)

    assert Xor(x, y) == Xor(x1, y1)
    assert Xor(x1, y) == Xor(x1, y1)
    assert Xor(x, y)._sympy_() == sympy.Xor(x1, y1)
    assert sympify(sympy.Xor(x1, y1)) == Xor(x, y)

    x = Symbol("x")
    x1 = sympy.Symbol("x")

    assert Piecewise((x, x < 1), (0, True)) == Piecewise((x1, x1 < 1), (0, True))
    assert Piecewise((x, x1 < 1), (0, True)) == Piecewise((x1, x1 < 1), (0, True))
    assert Piecewise((x, x < 1), (0, True))._sympy_() == sympy.Piecewise((x1, x1 < 1), (0, True))
    assert sympify(sympy.Piecewise((x1, x1 < 1), (0, True))) == Piecewise((x, x < 1), (0, True))

    assert Contains(x, Interval(1, 1)) == Contains(x1, Interval(1, 1))
    assert Contains(x, Interval(1, 1))._sympy_() == sympy.Contains(x1, Interval(1, 1))
    assert sympify(sympy.Contains(x1, Interval(1, 1))) == Contains(x, Interval(1, 1))


@unittest.skipIf(not have_sympy, "SymPy not installed")
def test_sets():
    x = Integer(2)
    y = Integer(3)
    x1 = sympy.Integer(2)
    y1 = sympy.Integer(3)

    assert Interval(x, y) == Interval(x1, y1)
    assert Interval(x1, y) == Interval(x1, y1)
    assert Interval(x, y)._sympy_() == sympy.Interval(x1, y1)
    assert sympify(sympy.Interval(x1, y1)) == Interval(x, y)

    assert sympify(sympy.S.EmptySet) == EmptySet()
    assert sympy.S.EmptySet == EmptySet()._sympy_()

    assert sympify(sympy.S.UniversalSet) == UniversalSet()
    assert sympy.S.UniversalSet == UniversalSet()._sympy_()

    assert sympify(sympy.S.Reals) == Reals()
    assert sympy.S.Reals == Reals()._sympy_()

    assert sympify(sympy.S.Rationals) == Rationals()
    assert sympy.S.Rationals == Rationals()._sympy_()

    assert sympify(sympy.S.Integers) == Integers()
    assert sympy.S.Integers == Integers()._sympy_()

    assert FiniteSet(x, y) == FiniteSet(x1, y1)
    assert FiniteSet(x1, y) == FiniteSet(x1, y1)
    assert FiniteSet(x, y)._sympy_() == sympy.FiniteSet(x1, y1)
    assert sympify(sympy.FiniteSet(x1, y1)) == FiniteSet(x, y)

    x = Interval(1, 2)
    y = Interval(2, 3)
    x1 = sympy.Interval(1, 2)
    y1 = sympy.Interval(2, 3)

    assert Union(x, y) == Union(x1, y1)
    assert Union(x1, y) == Union(x1, y1)
    assert Union(x, y)._sympy_() == sympy.Union(x1, y1)
    assert sympify(sympy.Union(x1, y1)) == Union(x, y)

    assert Complement(x, y) == Complement(x1, y1)
    assert Complement(x1, y) == Complement(x1, y1)
    assert Complement(x, y)._sympy_() == sympy.Complement(x1, y1)
    assert sympify(sympy.Complement(x1, y1)) == Complement(x, y)


@unittest.skipIf(not have_sympy, "SymPy not installed")
def test_pynumber():
    a = sympy.FF(7)(3)
    b = sympify(a)

    assert isinstance(b, PyNumber)

    a = a + 1
    b = b + 1
    assert isinstance(b, PyNumber)
    assert b == a                  # Check equality via SymEngine
    assert a == b                  # Check equality via SymPy
    assert str(a) == str(b)

    a = 1 - a
    b = 1 - b
    assert isinstance(b, PyNumber)
    assert b == a                  # Check equality via SymEngine
    assert a == b                  # Check equality via SymPy

    a = 2 * a
    b = 2 * b
    assert isinstance(b, PyNumber)
    assert b == a                  # Check equality via SymEngine
    assert a == b                  # Check equality via SymPy

    if sympy.__version__ != '1.2':
        a = 2 / a
        b = 2 / b
        assert isinstance(b, PyNumber)
        assert b == a                  # Check equality via SymEngine
        assert a == b                  # Check equality via SymPy

    x = Symbol("x")
    b = x * sympy.FF(7)(3)
    assert isinstance(b, Mul)

    b = b / x
    assert isinstance(b, PyNumber)


@unittest.skipIf(not have_sympy, "SymPy not installed")
def test_construct_dense_matrix():
    # Test for issue #347
    A = sympy.Matrix([[1, 2], [3, 5]])
    B = DenseMatrix(A)
    assert B.shape == (2, 2)
    assert list(B) == [1, 2, 3, 5]


@unittest.skipIf(not have_sympy, "SymPy not installed")
def test_conv_doubles():
    f = 4.347249999999999
    a = sympify(f)
    assert isinstance(a, RealDouble)
    assert sympify(a._sympy_()) == a
    assert float(a) == f
    assert float(a._sympy_()) == f

def test_conv_large_integers():
    a = Integer(10)**10000
    # check that convert to python int does not throw
    b = int(a)
    # check that convert to sympy int does not throw
    if have_sympy:
        c = a._sympy_()
        d = sympify(c)
