from symengine.test_utilities import raises
from symengine.lib.symengine_wrapper import (series, have_piranha, have_flint,
    Symbol, Integer, sin, cos, exp, sqrt, E)


def test_series_expansion():
    x = Symbol('x')
    ex = series(sin(1+x), x, n=10)
    assert ex.coeff(x, 7) == -cos(1)/5040

    x = Symbol('x')
    ex = series(1/(1-x), x, n=10)
    assert ex.coeff(x, 9) == 1
    ex = series(sin(x)*cos(x), x, n=10)
    assert ex.coeff(x, 8) == 0
    assert ex.coeff(x, 9) == Integer(2)/Integer(2835)

    ex = series(E**x, x, n=10)
    assert ex.coeff(x, 9) == Integer(1)/Integer(362880)
    ex1 = series(1/sqrt(4-x), x, n=50)
    ex2 = series((4-x)**(Integer(-1)/Integer(2)), x, n=50)
    assert ex1.coeff(x, 49) == ex2.coeff(x, 49)
