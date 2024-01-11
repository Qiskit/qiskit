from symengine.test_utilities import raises
from symengine.lib.symengine_wrapper import (true, false, Eq, Ne, Ge, Gt, Le, Lt, Symbol,
                                            I, And, Or, Not, Nand, Nor, Xor, Xnor, Piecewise,
                                            Contains, Interval, FiniteSet, oo, log)

x = Symbol("x")
y = Symbol("y")
z = Symbol("z")

def test_relationals():
    assert Eq(0) == true
    assert Eq(1) == false
    assert Eq(x, x) == true
    assert Eq(0, 0) == true
    assert Eq(1, 0) == false
    assert Ne(0, 0) == false
    assert Ne(1, 0) == true
    assert Lt(0, 1) == true
    assert Lt(1, 0) == false
    assert Le(0, 1) == true
    assert Le(1, 0) == false
    assert Le(0, 0) == true
    assert Gt(1, 0) == true
    assert Gt(0, 1) == false
    assert Ge(1, 0) == true
    assert Ge(0, 1) == false
    assert Ge(1, 1) == true
    assert Eq(I, 2) == false
    assert Ne(I, 2) == true


def test_rich_cmp():
    assert (x < y) == Lt(x, y)
    assert (x <= y) == Le(x, y)
    assert (x > y) == Gt(x, y)
    assert (x >= y) == Ge(x, y)


def test_And():
    assert And() == true
    assert And(True) == true
    assert And(False) == false
    assert And(True, True ) == true
    assert And(True, False) == false
    assert And(False, False) == false
    assert And(True, True, True) == true
    raises(TypeError, lambda: x < y and y < 1)


def test_Or():
    assert Or() == false
    assert Or(True) == true
    assert Or(False) == false
    assert Or(True, True ) == true
    assert Or(True, False) == true
    assert Or(False, False) == false
    assert Or(True, False, False) == true
    raises(TypeError, lambda: x < y or y < 1)


def test_Nor():
    assert Nor() == true
    assert Nor(True) == false
    assert Nor(False) == true
    assert Nor(True, True ) == false
    assert Nor(True, False) == false
    assert Nor(False, False) == true
    assert Nor(True, True, True) == false


def test_Nand():
    assert Nand() == false
    assert Nand(True) == false
    assert Nand(False) == true
    assert Nand(True, True) == false
    assert Nand(True, False) == true
    assert Nand(False, False) == true
    assert Nand(True, True, True) == false


def test_Not():
    assert Not(True) == false
    assert Not(False) == true


def test_Xor():
    assert Xor() == false
    assert Xor(True) == true
    assert Xor(False) == false
    assert Xor(True, True ) == false
    assert Xor(True, False) == true
    assert Xor(False, False) == false
    assert Xor(True, False, False) == true


def test_Xnor():
    assert Xnor() == true
    assert Xnor(True) == false
    assert Xnor(False) == true
    assert Xnor(True, True ) == true
    assert Xnor(True, False) == false
    assert Xnor(False, False) == true
    assert Xnor(True, False, False) == false


def test_Piecewise():
    assert Piecewise((x, x < 1), (0, True)) == Piecewise((x, x < 1), (0, True))
    int1 = Interval(1, 2, True, False)
    int2 = Interval(2, 5, True, False)
    int3 = Interval(5, 10, True, False)
    p = Piecewise((x, Contains(x, int1)), (y, Contains(x, int2)), (x + y, Contains(x, int3)))
    q = Piecewise((1, Contains(x, int1)), (0, Contains(x, int2)), (1, Contains(x, int3)))
    assert p.diff(x) == q


def test_Contains():
    assert Contains(x, FiniteSet(0)) != false
    assert Contains(x, Interval(1, 1)) != false
    assert Contains(oo, Interval(-oo, oo)) == false
    assert Contains(-oo, Interval(-oo, oo)) == false

