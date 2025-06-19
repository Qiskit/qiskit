from symengine.test_utilities import raises
from symengine.lib.symengine_wrapper import (Interval, EmptySet, UniversalSet,
    FiniteSet, Union, Complement, ImageSet, ConditionSet, Reals, Rationals,
    Integers, And, Or, oo, Symbol, true, Ge, Eq, Gt)


def test_Interval():
    assert Interval(0, oo) == Interval(0, oo, False, True)
    assert Interval(0, oo) == Interval(0, oo, left_open=False, right_open=True)
    assert Interval(-oo, 0) == Interval(-oo, 0, True, False)
    assert Interval(-oo, 0) == Interval(-oo, 0, left_open=True, right_open=False)
    assert Interval(oo, -oo) == EmptySet()
    assert Interval(oo, oo) == EmptySet()
    assert Interval(-oo, -oo) == EmptySet()
    assert isinstance(Interval(1, 1), FiniteSet)

    assert Interval(1, 0) == EmptySet()
    assert Interval(1, 1, False, True) == EmptySet()
    assert Interval(1, 1, True, False) == EmptySet()
    assert Interval(1, 1, True, True) == EmptySet()
    assert Interval(1, 2).union(Interval(2, 3)) == Interval(1, 3)

    assert Interval(-oo, 0).start == -oo
    assert Interval(-oo, 0).end == 0


def test_EmptySet():
    E = EmptySet()
    assert E.intersection(UniversalSet()) == E


def test_UniversalSet():
    U = UniversalSet()
    x = Symbol("x")
    assert U.union(Interval(2, 4)) == U
    assert U.intersection(Interval(2, 4)) == Interval(2, 4)
    assert U.contains(0) == true


def test_Reals():
    R = Reals()
    assert R.union(Interval(2, 4)) == R
    assert R.contains(0) == true


def test_Rationals():
    Q = Rationals()
    assert Q.union(FiniteSet(2, 3)) == Q
    assert Q.contains(0) == true


def test_Integers():
    Z = Integers()
    assert Z.union(FiniteSet(2, 4)) == Z
    assert Z.contains(0) == true


def test_FiniteSet():
    x = Symbol("x")
    A = FiniteSet(1, 2, 3)
    B = FiniteSet(3, 4, 5)
    AorB = Union(A, B)
    AandB = A.intersection(B)
    assert AandB == FiniteSet(3)

    assert FiniteSet(EmptySet()) != EmptySet()
    assert FiniteSet(FiniteSet(1, 2, 3)) != FiniteSet(1, 2, 3)


def test_Union():
    assert Union(Interval(1, 2), Interval(2, 3)) == Interval(1, 3)
    assert Union(Interval(1, 2), Interval(2, 3, True)) == Interval(1, 3)
    assert Union(Interval(1, 3), Interval(2, 4)) == Interval(1, 4)
    assert Union(Interval(1, 2), Interval(1, 3)) == Interval(1, 3)
    assert Union(Interval(1, 3), Interval(1, 2)) == Interval(1, 3)
    assert Union(Interval(1, 3, False, True), Interval(1, 2)) == \
        Interval(1, 3, False, True)
    assert Union(Interval(1, 3), Interval(1, 2, False, True)) == Interval(1, 3)
    assert Union(Interval(1, 2, True), Interval(1, 3)) == Interval(1, 3)
    assert Union(Interval(1, 2, True), Interval(1, 3, True)) == \
        Interval(1, 3, True)
    assert Union(Interval(1, 2, True), Interval(1, 3, True, True)) == \
        Interval(1, 3, True, True)
    assert Union(Interval(1, 2, True, True), Interval(1, 3, True)) == \
        Interval(1, 3, True)
    assert Union(Interval(1, 3), Interval(2, 3)) == Interval(1, 3)
    assert Union(Interval(1, 3, False, True), Interval(2, 3)) == \
        Interval(1, 3)
    assert Union(Interval(1, 2, False, True), Interval(2, 3, True)) != \
        Interval(1, 3)
    assert Union(Interval(1, 2), EmptySet()) == Interval(1, 2)
    assert Union(EmptySet()) == EmptySet()


def test_Complement():
    assert Complement(Interval(1, 3), Interval(1, 2)) == Interval(2, 3, True)
    assert Complement(FiniteSet(1, 3, 4), FiniteSet(3, 4)) == FiniteSet(1)
    assert Complement(Union(Interval(0, 2),
                            FiniteSet(2, 3, 4)), Interval(1, 3)) == \
        Union(Interval(0, 1, False, True), FiniteSet(4))


def test_ConditionSet():
    x = Symbol("x")
    i1 = Interval(-oo, oo)
    f1 = FiniteSet(0, 1, 2, 4)
    cond1 = Ge(x**2, 9)
    assert ConditionSet(x, And(Eq(0, 1), i1.contains(x))) == EmptySet()
    assert ConditionSet(x, And(Gt(1, 0), i1.contains(x))) == i1
    assert ConditionSet(x, And(cond1, f1.contains(x))) == FiniteSet(4)


def test_ImageSet():
    x = Symbol("x")
    i1 = Interval(0, 1)
    assert ImageSet(x, x**2, EmptySet()) == EmptySet()
    assert ImageSet(x, 1, i1) == FiniteSet(1)
    assert ImageSet(x, x, i1) == i1
