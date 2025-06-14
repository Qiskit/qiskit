from symengine import Symbol, symbols, symarray, has_symbol, Dummy
from symengine.test_utilities import raises
import unittest
import platform


def test_symbol():
    x = Symbol("x")
    assert x.name == "x"
    assert str(x) == "x"
    assert str(x) != "y"
    assert repr(x) == str(x)


def test_symbols():
    x = Symbol('x')
    y = Symbol('y')
    z = Symbol('z')

    assert symbols('x') == x
    assert symbols('x ') == x
    assert symbols(' x ') == x
    assert symbols('x,') == (x,)
    assert symbols('x, ') == (x,)
    assert symbols('x ,') == (x,)

    assert symbols('x , y') == (x, y)

    assert symbols('x,y,z') == (x, y, z)
    assert symbols('x y z') == (x, y, z)

    assert symbols('x,y,z,') == (x, y, z)
    assert symbols('x y z ') == (x, y, z)

    xyz = Symbol('xyz')
    abc = Symbol('abc')

    assert symbols('xyz') == xyz
    assert symbols('xyz,') == (xyz,)
    assert symbols('xyz,abc') == (xyz, abc)

    assert symbols(('xyz',)) == (xyz,)
    assert symbols(('xyz,',)) == ((xyz,),)
    assert symbols(('x,y,z,',)) == ((x, y, z),)
    assert symbols(('xyz', 'abc')) == (xyz, abc)
    assert symbols(('xyz,abc',)) == ((xyz, abc),)
    assert symbols(('xyz,abc', 'x,y,z')) == ((xyz, abc), (x, y, z))

    assert symbols(('x', 'y', 'z')) == (x, y, z)
    assert symbols(['x', 'y', 'z']) == [x, y, z]
    assert symbols({'x', 'y', 'z'}) == {x, y, z}

    raises(ValueError, lambda: symbols(''))
    raises(ValueError, lambda: symbols(','))
    raises(ValueError, lambda: symbols('x,,y,,z'))
    raises(ValueError, lambda: symbols(('x', '', 'y', '', 'z')))

    x0 = Symbol('x0')
    x1 = Symbol('x1')
    x2 = Symbol('x2')

    y0 = Symbol('y0')
    y1 = Symbol('y1')

    assert symbols('x0:0') == ()
    assert symbols('x0:1') == (x0,)
    assert symbols('x0:2') == (x0, x1)
    assert symbols('x0:3') == (x0, x1, x2)

    assert symbols('x:0') == ()
    assert symbols('x:1') == (x0,)
    assert symbols('x:2') == (x0, x1)
    assert symbols('x:3') == (x0, x1, x2)

    assert symbols('x1:1') == ()
    assert symbols('x1:2') == (x1,)
    assert symbols('x1:3') == (x1, x2)

    assert symbols('x1:3,x,y,z') == (x1, x2, x, y, z)

    assert symbols('x:3,y:2') == (x0, x1, x2, y0, y1)
    assert symbols(('x:3', 'y:2')) == ((x0, x1, x2), (y0, y1))

    a = Symbol('a')
    b = Symbol('b')
    c = Symbol('c')
    d = Symbol('d')

    assert symbols('x:z') == (x, y, z)
    assert symbols('a:d,x:z') == (a, b, c, d, x, y, z)
    assert symbols(('a:d', 'x:z')) == ((a, b, c, d), (x, y, z))

    aa = Symbol('aa')
    ab = Symbol('ab')
    ac = Symbol('ac')
    ad = Symbol('ad')

    assert symbols('aa:d') == (aa, ab, ac, ad)
    assert symbols('aa:d,x:z') == (aa, ab, ac, ad, x, y, z)
    assert symbols(('aa:d', 'x:z')) == ((aa, ab, ac, ad), (x, y, z))

    def sym(s):
        return str(symbols(s))
    assert sym('a0:4') == '(a0, a1, a2, a3)'
    assert sym('a2:4,b1:3') == '(a2, a3, b1, b2)'
    assert sym('a1(2:4)') == '(a12, a13)'
    assert sym('a0:2.0:2') == '(a0.0, a0.1, a1.0, a1.1)'
    assert sym('aa:cz') == '(aaz, abz, acz)'
    assert sym('aa:c0:2') == '(aa0, aa1, ab0, ab1, ac0, ac1)'
    assert sym('aa:ba:b') == '(aaa, aab, aba, abb)'
    assert sym('a:3b') == '(a0b, a1b, a2b)'
    assert sym('a-1:3b') == '(a-1b, a-2b)'
    assert sym(r'a:2\,:2' + chr(0)) == '(a0,0%s, a0,1%s, a1,0%s, a1,1%s)' % (
        (chr(0),)*4)
    assert sym('x(:a:3)') == '(x(a0), x(a1), x(a2))'
    assert sym('x(:c):1') == '(xa0, xb0, xc0)'
    assert sym('x((:a)):3') == '(x(a)0, x(a)1, x(a)2)'
    assert sym('x(:a:3') == '(x(a0, x(a1, x(a2)'
    assert sym(':2') == '(0, 1)'
    assert sym(':b') == '(a, b)'
    assert sym(':b:2') == '(a0, a1, b0, b1)'
    assert sym(':2:2') == '(00, 01, 10, 11)'
    assert sym(':b:b') == '(aa, ab, ba, bb)'

    raises(ValueError, lambda: symbols(':'))
    raises(ValueError, lambda: symbols('a:'))
    raises(ValueError, lambda: symbols('::'))
    raises(ValueError, lambda: symbols('a::'))
    raises(ValueError, lambda: symbols(':a:'))
    raises(ValueError, lambda: symbols('::a'))


def test_symarray():
    try:
        import numpy as np
    except ImportError:
        return
    x0, x1, x2 = arr = symarray('x', 3)
    assert arr.shape == (3,)

    assert symarray('y', (2, 3)).shape == (2, 3)


def test_has_symbol():
    a = Symbol('a')
    b = Symbol('b')
    c = Symbol('c')

    assert not has_symbol(2, a)
    assert not has_symbol(c, a)
    assert has_symbol(a+b, b)


def test_dummy():
    x1 = Symbol('x')
    x2 = Symbol('x')
    xdummy1 = Dummy('x')
    xdummy2 = Dummy('x')

    assert x1 == x2
    assert x1 != xdummy1
    assert xdummy1 == (xdummy1 + 1) - 1
    assert xdummy1 != xdummy2
    assert Dummy() != Dummy()
    assert Dummy('x') != Dummy('x')

# Cython cdef classes on PyPy has a __dict__ attribute always
# __slots__ on PyPy are useless anyways. https://stackoverflow.com/a/23077685/4768820
@unittest.skipUnless(platform.python_implementation()=="CPython", "__slots__ are useless on PyPy")
def test_slots():
    x = Dummy('x')
    # Verify the successful use of slots.
    assert not hasattr(x, "__dict__")
    assert not hasattr(x, "__weakref__")

    x1 = Symbol('x')
    # Verify the successful use of slots.
    assert not hasattr(x, "__dict__")
    assert not hasattr(x, "__weakref__")
