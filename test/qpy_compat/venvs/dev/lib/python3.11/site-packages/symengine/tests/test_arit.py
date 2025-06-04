from symengine.test_utilities import raises

from symengine import (Symbol, Integer, Add, Mul, Pow, Rational, sqrt,
    symbols, S, I, count_ops, floor)


def test_arit1():
    x = Symbol("x")
    y = Symbol("y")
    e = x + y
    e = x * y
    e = Integer(2)*x
    e = 2*x
    e = x + 1
    e = 1 + x


def test_arit2():
    x = Symbol("x")
    y = Symbol("y")
    assert x+x == Integer(2) * x
    assert x+x != Integer(3) * x
    assert x+y == y+x
    assert x+x == 2*x
    assert x+x == x*2
    assert x+x+x == 3*x
    assert x+y+x+x == 3*x+y

    assert not x+x == 3*x
    assert not x+x != 2*x


def test_arit3():
    x = Symbol("x")
    y = Symbol("y")
    raises(TypeError, lambda: ("x"*x))


def test_arit4():
    x = Symbol("x")
    y = Symbol("y")
    assert x*x == x**2
    assert x*y == y*x
    assert x*x*x == x**3
    assert x*y*x*x == x**3*y


def test_arit5():
    x = Symbol("x")
    y = Symbol("y")
    e = (x+y)**2
    f = e.expand()
    assert e == (x+y)**2
    assert e != x**2 + 2*x*y + y**2
    assert isinstance(e, Pow)
    assert f == x**2 + 2*x*y + y**2
    assert isinstance(f, Add)


def test_arit6():
    x = Symbol("x")
    y = Symbol("y")
    e = x + y
    assert str(e) == "x + y" or "y + x"
    e = x * y
    assert str(e) == "x*y" or "y*x"
    e = Integer(2)*x
    assert str(e) == "2*x"
    e = 2*x
    assert str(e) == "2*x"


def test_arit7():
    x = Symbol("x")
    y = Symbol("y")
    assert x - x == 0
    assert x - y != y - x
    assert 2*x - x == x
    assert 3*x - x == 2*x

    assert 2*x*y - x*y == x*y


def test_arit8():
    x = Symbol("x")
    y = Symbol("y")
    z = Symbol("z")
    assert x**y * x**x == x**(x+y)
    assert x**y * x**x * x**z == x**(x+y+z)
    assert x**y - x**y == 0

    assert x**2 / x == x
    assert y*x**2 / (x*y) == x
    assert (2 * x**3 * y**2 * z)**3 / 8 == x**9 * y**6 * z**3
    assert (2*y**(-2*x**2)) * (3*y**(2*x**2)) == 6


def test_unary():
    x = Symbol("x")
    assert -x == 0 - x
    assert +x == x


def test_expand1():
    x = Symbol("x")
    y = Symbol("y")
    z = Symbol("z")
    assert ((2*x+y)**2).expand() == 4*x**2 + 4*x*y + y**2
    assert (x**2)**3 == x**6
    assert ((2*x**2+3*y)**2).expand() == 4*x**4 + 12*x**2*y + 9*y**2
    assert ((2*x/3+y/4)**2).expand() == 4*x**2/9 + x*y/3 + y**2/16


def test_arit9():
    x = Symbol("x")
    y = Symbol("y")
    assert 1/x == 1/x
    assert 1/x != 1/y


def test_expand2():
    x = Symbol("x")
    y = Symbol("y")
    z = Symbol("z")
    assert ((1/(y*z) - y*z)*y*z).expand() == 1-(y*z)**2
    assert (2*(x + 2*(y + z))).expand(deep=False) == 2*x + 4*(y+z)
    ex = x + 2*(y + z)
    assert ex.expand(deep=False) == ex


def test_expand3():
    x = Symbol("x")
    y = Symbol("y")
    assert ((1/(x*y) - x*y+2)*(1+x*y)).expand() == 3 + 1/(x*y) + x*y - (x*y)**2


def test_args():
    x = Symbol("x")
    y = Symbol("y")
    assert (x**2).args == (x, 2)
    assert (x**2 + 5).args == (5, x**2)
    assert set((x**2 + 2*x*y + 5).args) == {x**2, 2*x*y, Integer(5)}
    assert (2*x**2).args == (2, x**2)
    assert set((2*x**2*y).args) == {Integer(2), x**2, y}


def test_atoms():
    x = Symbol("x")
    y = Symbol("y")
    z = Symbol("z")
    assert (x**2).atoms() == {x}
    assert (x**2).atoms(Symbol) == {x}
    assert (x ** y + z).atoms() == {x, y, z}
    assert (x**y + z).atoms(Symbol) == {x, y, z}


def test_free_symbols():
    x = Symbol("x")
    y = Symbol("y")
    z = Symbol("z")
    assert (x**2).free_symbols == {x}
    assert (x**y + z).free_symbols == {x, y, z}


def test_as_numer_denom():
    x, y = Rational(17, 26).as_numer_denom()
    assert x == Integer(17)
    assert y == Integer(26)

    x, y = Integer(-5).as_numer_denom()
    assert x == Integer(-5)
    assert y == Integer(1)


def test_floor():
    exprs = [Symbol("x"), Symbol("y"), Integer(2), Rational(-3, 5), Integer(-3)]

    for x in exprs:
        for y in exprs:
            assert x // y == floor(x / y)
            assert x == y * (x // y) + x % y


def test_as_real_imag():
    x, y = (5 + 6 * I).as_real_imag()

    assert x == 5
    assert y == 6


def test_from_args():
    x = Symbol("x")
    y = Symbol("y")
    
    assert Add._from_args([]) == 0
    assert Add._from_args([x]) == x
    assert Add._from_args([x, y]) == x + y

    assert Mul._from_args([]) == 1
    assert Mul._from_args([x]) == x
    assert Mul._from_args([x, y]) == x * y


def test_make_args():
    x = Symbol("x")
    y = Symbol("y")
    z = Symbol("z")

    assert Add.make_args(x) == (x,)
    assert Mul.make_args(x) == (x,)

    assert Add.make_args(x*y*z) == (x*y*z,)
    assert Mul.make_args(x*y*z) == (x*y*z).args

    assert Add.make_args(x + y + z) == (x + y + z).args
    assert Mul.make_args(x + y + z) == (x + y + z,)

    assert Add.make_args((x + y)**z) == ((x + y)**z,)
    assert Mul.make_args((x + y)**z) == ((x + y)**z,)


def test_Pow_base_exp():
    x = Symbol("x")
    y = Symbol("y")
    e = Pow(x + y, 2)
    assert isinstance(e, Pow)
    assert e.exp == 2
    assert e.base == x + y

    assert sqrt(x - 1).as_base_exp() == (x - 1, Rational(1, 2))


def test_copy():
    b = Symbol("b")
    a = b.copy()
    assert a is b
    assert type(a) == type(b)


def test_special_constants():
    assert S.Zero == Integer(0)
    assert S.One == Integer(1)
    assert S.NegativeOne == Integer(-1)
    assert S.Half == Rational(1, 2)


def test_bool():
    x = Symbol('x')
    if (x**2).args[1] > 0:
        assert True
    if (x**2).args[1] < 0:
        assert False


def test_count_ops():
    x, y = symbols("x, y")
    assert count_ops(x+y) == 1
    assert count_ops((x+y, x*y)) == 2
    assert count_ops([[x**y], [x+y-1]]) == 3
    assert count_ops(x+y, x*y) == 2

