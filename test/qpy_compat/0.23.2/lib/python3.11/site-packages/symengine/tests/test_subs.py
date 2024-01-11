import unittest

from symengine.test_utilities import raises
from symengine import Symbol, sin, cos, sqrt, Add, function_symbol, have_numpy


def test_basic():
    x = Symbol("x")
    y = Symbol("y")
    z = Symbol("z")
    e = x+y+z
    assert e.subs({x: y, z: y}) == 3*y


def test_sin():
    x = Symbol("x")
    y = Symbol("y")
    e = sin(x)
    assert e.subs({x: y}) == sin(y)
    assert e.subs({x: y}) != sin(x)

    e = cos(x)
    assert e.subs({x: 0}) == 1
    assert e.subs(x, 0) == 1


def test_args():
    x = Symbol("x")
    e = cos(x)
    raises(TypeError, lambda: e.subs(x, 0, 3))


def test_f():
    x = Symbol("x")
    y = Symbol("y")
    f = function_symbol("f", x)
    g = function_symbol("g", x)
    assert f.subs({function_symbol("f", x): function_symbol("g", x)}) == g
    assert ((f+g).subs({function_symbol("f", x): function_symbol("g", x)}) ==
            2*g)

    e = (f+x)**3
    assert e.subs({f: y}) == (x+y)**3
    e = e.expand()
    assert e.subs({f: y}) == ((x+y)**3).expand()


def test_msubs():
    x = Symbol("x")
    y = Symbol("y")
    f = function_symbol("f", x)
    assert f.msubs({f: y}) == y
    assert f.diff(x).msubs({f: y}) == f.diff(x)


def test_xreplace():
    x = Symbol("x")
    y = Symbol("y")
    f = sin(cos(x))
    assert f.xreplace({x: y}) == sin(cos(y))


@unittest.skipUnless(have_numpy, "Numpy not installed")
def test_float32():
    import numpy as np
    x = Symbol("x")
    expr = x * 2
    assert expr.subs({x: np.float32(2)}) == 4.0


@unittest.skipUnless(have_numpy, "Numpy not installed")
def test_float16():
    import numpy as np
    x = Symbol("x")
    expr = x * 2
    assert expr.subs({x: np.float16(2)}) == 4.0
