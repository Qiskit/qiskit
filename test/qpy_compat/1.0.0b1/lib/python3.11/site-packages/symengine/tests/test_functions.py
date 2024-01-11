from symengine import (
    Symbol, sin, cos, sqrt, Add, Mul, function_symbol, Integer, log, E, symbols, I,
    Rational, EulerGamma, Function, Subs, Derivative, LambertW, zeta, dirichlet_eta,
    zoo, pi, KroneckerDelta, LeviCivita, erf, erfc, oo, lowergamma, uppergamma, exp,
    loggamma, beta, polygamma, digamma, trigamma, sign, floor, ceiling, conjugate,
    nan, Float, UnevaluatedExpr
)
from symengine.test_utilities import raises

import unittest

try:
    import sympy
    from sympy.core.cache import clear_cache
    import atexit
    atexit.register(clear_cache)
    have_sympy = True
except ImportError:
    have_sympy = False

def test_sin():
    x = Symbol("x")
    e = sin(x)
    assert e == sin(x)
    assert e != cos(x)

    assert sin(x).diff(x) == cos(x)
    assert cos(x).diff(x) == -sin(x)

    e = sqrt(x).diff(x).diff(x)
    f = sin(e)
    g = f.diff(x).diff(x)
    assert isinstance(g, Add)


def test_f():
    x = Symbol("x")
    y = Symbol("y")
    z = Symbol("z")
    f = function_symbol("f", x)
    g = function_symbol("g", x)
    assert f != g

    f = function_symbol("f", x)
    g = function_symbol("f", x)
    assert f == g

    f = function_symbol("f", x, y)
    g = function_symbol("f", y, x)
    assert f != g

    f = function_symbol("f", x, y)
    g = function_symbol("f", x, y)
    assert f == g


def test_derivative():
    x = Symbol("x")
    y = Symbol("y")
    f = function_symbol("f", x)
    assert f.diff(x) == function_symbol("f", x).diff(x)
    assert f.diff(x).diff(x) == function_symbol("f", x).diff(x).diff(x)
    assert f.diff(y) == 0
    assert f.diff(x).args == (f, x)
    assert f.diff(x).diff(x).args == (f, x, x)
    assert f.diff(x, 0) == f
    assert f.diff(x, 0) == Derivative(function_symbol("f", x), x, 0)
    raises(ValueError, lambda: f.diff(0))
    raises(ValueError, lambda: f.diff(x, 0, 0))
    raises(ValueError, lambda: f.diff(x, y, 0, 0, x))

    g = function_symbol("f", y)
    assert g.diff(x) == 0
    assert g.diff(y) == function_symbol("f", y).diff(y)
    assert g.diff(y).diff(y) == function_symbol("f", y).diff(y).diff(y)

    assert f - function_symbol("f", x) == 0

    f = function_symbol("f", x, y)
    assert f.diff(x).diff(y) == function_symbol("f", x, y).diff(x).diff(y)
    assert f.diff(Symbol("z")) == 0

    s = Derivative(function_symbol("f", x), x)
    assert s.expr == function_symbol("f", x)
    assert s.variables == (x,)

    fxy = Function("f")(x, y)
    assert (1+fxy).has(fxy)
    g = Derivative(Function("f")(x, y), x, 2, y, 1)
    assert g == fxy.diff(x, x, y)
    assert g == fxy.diff(y, 1, x, 2)
    assert g == fxy.diff(y, x, 2)

    h = Derivative(Function("f")(x, y), x, 0, y, 1)
    assert h == fxy.diff(x, 0, y)
    assert h == fxy.diff(y, x, 0)

    i = Derivative(Function("f")(x, y), x, 0, y, 1, x, 1)
    assert i == fxy.diff(x, 0, y, x, 1)
    assert i == fxy.diff(x, 0, y, x)
    assert i == fxy.diff(y, x)
    assert i == fxy.diff(y, 1, x, 1)
    assert i == fxy.diff(y, 1, x)


def test_abs():
    x = Symbol("x")
    e = abs(x)
    assert e == abs(x)
    assert e != cos(x)

    assert abs(5) == 5
    assert abs(-5) == 5
    assert abs(Integer(5)/3) == Integer(5)/3
    assert abs(-Integer(5)/3) == Integer(5)/3
    assert abs(Integer(5)/3+x) != Integer(5)/3
    assert abs(Integer(5)/3+x) == abs(Integer(5)/3+x)


def test_abs_diff():
    x = Symbol("x")
    y = Symbol("y")
    e = abs(x)
    assert e.diff(x) != e
    assert e.diff(x) != 0
    assert e.diff(y) == 0


def test_Subs():
    x = Symbol("x")
    y = Symbol("y")
    _x = Symbol("_xi_1")
    f = function_symbol("f", 2*x)
    assert str(f.diff(x)) == "2*Subs(Derivative(f(_xi_1), _xi_1), (_xi_1), (2*x))"
    # TODO: fix me
    # assert f.diff(x) == 2 * Subs(Derivative(function_symbol("f", _x), _x), [_x], [2 * x])
    assert Subs(Derivative(function_symbol("f", x, y), x), [x, y], [_x, x]) \
                == Subs(Derivative(function_symbol("f", x, y), x), [y, x], [x, _x])

    s = f.diff(x)/2
    _xi_1 = Symbol("_xi_1")
    assert s.expr == Derivative(function_symbol("f", _xi_1), _xi_1)
    assert s.variables == (_xi_1,)
    assert s.point == (2*x,)


@unittest.skipUnless(have_sympy, "SymPy not installed")
def test_FunctionWrapper():
    import sympy
    n, m, theta, phi = sympy.symbols("n, m, theta, phi")
    r = sympy.Ynm(n, m, theta, phi)
    s = Integer(2)*r
    assert isinstance(s, Mul)
    assert isinstance(s.args[1]._sympy_(), sympy.Ynm)

    x = symbols("x")
    e = x + sympy.Mod(x, 2)
    assert str(e) == "x + Mod(x, 2)"
    assert isinstance(e, Add)
    assert e + sympy.Mod(x, 2) == x + 2*sympy.Mod(x, 2)

    f = e.subs({x : 10})
    assert f == 10

    f = e.subs({x : 2})
    assert f == 2

    f = e.subs({x : 100});
    v = f.n(53, real=True);
    assert abs(float(v) - 100.00000000) < 1e-7


def test_log():
    x = Symbol("x")
    y = Symbol("y")
    assert log(E) == 1
    assert log(x, x) == 1
    assert log(x, y) == log(x) / log(y)


def test_lambertw():
    assert LambertW(0) == 0
    assert LambertW(E) == 1
    assert LambertW(-1/E) == -1
    assert LambertW(-log(2)/2) == -log(2)


def test_zeta():
    x = Symbol("x")
    assert zeta(1) == zoo
    assert zeta(1, x) == zoo

    assert zeta(0) == Rational(-1, 2)
    assert zeta(0, x) == Rational(1, 2) - x

    assert zeta(1, 2) == zoo
    assert zeta(1, -7) == zoo

    assert zeta(2, 1) == pi**2/6

    assert zeta(2) == pi**2/6
    assert zeta(4) == pi**4/90
    assert zeta(6) == pi**6/945

    assert zeta(2, 2) == pi**2/6 - 1
    assert zeta(4, 3) == pi**4/90 - Rational(17, 16)
    assert zeta(6, 4) == pi**6/945 - Rational(47449, 46656)

    assert zeta(-1) == -Rational(1, 12)
    assert zeta(-2) == 0
    assert zeta(-3) == Rational(1, 120)
    assert zeta(-4) == 0
    assert zeta(-5) == -Rational(1, 252)

    assert zeta(-1, 3) == -Rational(37, 12)
    assert zeta(-1, 7) == -Rational(253, 12)
    assert zeta(-1, -4) == Rational(119, 12)
    assert zeta(-1, -9) == Rational(539, 12)

    assert zeta(-4, 3) == -17
    assert zeta(-4, -8) == 8772

    assert zeta(0, 1) == -Rational(1, 2)
    assert zeta(0, -1) == Rational(3, 2)

    assert zeta(0, 2) == -Rational(3, 2)
    assert zeta(0, -2) == Rational(5, 2)


def test_dirichlet_eta():
    assert dirichlet_eta(0) == Rational(1, 2)
    assert dirichlet_eta(-1) == Rational(1, 4)
    assert dirichlet_eta(1) == log(2)
    assert dirichlet_eta(2) == pi**2/12
    assert dirichlet_eta(4) == pi**4*Rational(7, 720)


def test_kronecker_delta():
    x = Symbol("x")
    y = Symbol("y")
    assert KroneckerDelta(1, 1) == 1
    assert KroneckerDelta(1, 2) == 0
    assert KroneckerDelta(x, x) == 1
    assert KroneckerDelta(x**2 - y**2, x**2 - y**2) == 1
    assert KroneckerDelta(0, 0) == 1
    assert KroneckerDelta(0, 1) == 0


def test_levi_civita():
    i = Symbol("i")
    j = Symbol("j")
    assert LeviCivita(1, 2, 3) == 1
    assert LeviCivita(1, 3, 2) == -1
    assert LeviCivita(1, 2, 2) == 0
    assert LeviCivita(i, j, i) == 0
    assert LeviCivita(1, i, i) == 0
    assert LeviCivita(1, 2, 3, 1) == 0
    assert LeviCivita(4, 5, 1, 2, 3) == 1
    assert LeviCivita(4, 5, 2, 1, 3) == -1


def test_erf():
    x = Symbol("x")
    y = Symbol("y")
    assert erf(nan) == nan
    assert erf(oo) == 1
    assert erf(-oo) == -1
    assert erf(0) == 0
    assert erf(-2) == -erf(2)
    assert erf(-x*y) == -erf(x*y)
    assert erf(-x - y) == -erf(x + y)


def test_erfc():
    assert erfc(nan) == nan
    assert erfc(oo) == 0
    assert erfc(-oo) == 2
    assert erfc(0) == 1


def test_lowergamma():
    assert lowergamma(1, 2) == 1 - exp(-2)


def test_uppergamma():
    assert uppergamma(1, 2) == exp(-2)
    assert uppergamma(4, 0) == 6


def test_loggamma():
    assert loggamma(-1) == oo
    assert loggamma(-2) == oo
    assert loggamma(0) == oo
    assert loggamma(1) == 0
    assert loggamma(2) == 0
    assert loggamma(3) == log(2)


def test_beta():
    assert beta(3, 2) == beta(2, 3)


def test_polygamma():
    assert polygamma(0, -9) == zoo
    assert polygamma(0, -9) == zoo
    assert polygamma(0, -1) == zoo
    assert polygamma(0, 0) == zoo
    assert polygamma(0, 1) == -EulerGamma
    assert polygamma(0, 7) == Rational(49, 20) - EulerGamma
    assert polygamma(1, 1) == pi**2/6
    assert polygamma(1, 2) == pi**2/6 - 1
    assert polygamma(1, 3) == pi**2/6 - Rational(5, 4)
    assert polygamma(3, 1) == pi**4 / 15
    assert polygamma(3, 5) == 6*(Rational(-22369, 20736) + pi**4/90)
    assert polygamma(5, 1) == 8 * pi**6 / 63


def test_digamma():
    x = Symbol("x")
    assert digamma(x) == polygamma(0, x)
    assert digamma(0) == zoo
    assert digamma(1) == -EulerGamma


def test_trigamma():
    x = Symbol("x")
    assert trigamma(-2) == zoo
    assert trigamma(x) == polygamma(1, x)


def test_sign():
    assert sign(1.2) == 1
    assert sign(-1.2) == -1
    assert sign(3*I) == I
    assert sign(-3*I) == -I
    assert sign(0) == 0
    assert sign(nan) == nan


def test_floor():
    x = Symbol("x")
    y = Symbol("y")
    assert floor(nan) == nan
    assert floor(oo) == oo
    assert floor(-oo) == -oo
    assert floor(0) == 0
    assert floor(1) == 1
    assert floor(-1) == -1
    assert floor(E) == 2
    assert floor(pi) == 3
    assert floor(Rational(1, 2)) == 0
    assert floor(-Rational(1, 2)) == -1
    assert floor(Rational(7, 3)) == 2
    assert floor(-Rational(7, 3)) == -3
    assert floor(Float(17.0)) == 17
    assert floor(-Float(17.0)) == -17
    assert floor(Float(7.69)) == 7
    assert floor(-Float(7.69)) == -8
    assert floor(I) == I
    assert floor(-I) == -I
    assert floor(2*I) == 2*I
    assert floor(-2*I) == -2*I
    assert floor(E + pi) == floor(E + pi)
    assert floor(I + pi) == floor(I + pi)
    assert floor(floor(pi)) == 3
    assert floor(floor(y)) == floor(y)
    assert floor(floor(x)) == floor(floor(x))
    assert floor(x) == floor(x)
    assert floor(2*x) == floor(2*x)


def test_ceiling():
    x = Symbol("x")
    y = Symbol("y")
    assert ceiling(nan) == nan
    assert ceiling(oo) == oo
    assert ceiling(-oo) == -oo
    assert ceiling(0) == 0
    assert ceiling(1) == 1
    assert ceiling(-1) == -1
    assert ceiling(E) == 3
    assert ceiling(pi) == 4
    assert ceiling(Rational(1, 2)) == 1
    assert ceiling(-Rational(1, 2)) == 0
    assert ceiling(Rational(7, 3)) == 3
    assert ceiling(-Rational(7, 3)) == -2
    assert ceiling(Float(17.0)) == 17
    assert ceiling(-Float(17.0)) == -17
    assert ceiling(Float(7.69)) == 8
    assert ceiling(-Float(7.69)) == -7
    assert ceiling(I) == I
    assert ceiling(-I) == -I
    assert ceiling(2*I) == 2*I
    assert ceiling(-2*I) == -2*I
    assert ceiling(E + pi) == ceiling(E + pi)
    assert ceiling(I + pi) == ceiling(I + pi)
    assert ceiling(ceiling(pi)) == 4
    assert ceiling(ceiling(y)) == ceiling(y)
    assert ceiling(ceiling(x)) == ceiling(ceiling(x))
    assert ceiling(x) == ceiling(x)
    assert ceiling(2*x) == ceiling(2*x)


def test_conjugate():
    assert conjugate(pi) == pi
    assert conjugate(I) == -I


def test_unevaluated_expr():
    x = Symbol("x")
    t = UnevaluatedExpr(x)
    assert x + t != 2 * x
    assert not t.is_number
    assert not t.is_integer
    assert not t.is_finite

    t = UnevaluatedExpr(1)
    assert t.is_number
    assert t.is_integer
    assert t.is_finite
