from symengine import (ccode, unicode, Symbol, sqrt, Pow, Max, sin, Integer, MutableDenseMatrix)
from symengine.test_utilities import raises
from symengine.printing import CCodePrinter, init_printing

def test_ccode():
    x = Symbol("x")
    y = Symbol("y")
    assert ccode(x) == "x"
    assert ccode(x**3) == "pow(x, 3)"
    assert ccode(x**(y**3)) == "pow(x, pow(y, 3))"
    assert ccode(x**-1.0) == "pow(x, -1.0)"
    assert ccode(Max(x, x*x)) == "fmax(x, pow(x, 2))"
    assert ccode(sin(x)) == "sin(x)"
    assert ccode(Integer(67)) == "67"
    assert ccode(Integer(-1)) == "-1"

def test_CCodePrinter():
    x = Symbol("x")
    y = Symbol("y")
    myprinter = CCodePrinter()

    assert myprinter.doprint(1+x, "bork") == "bork = 1 + x;"
    assert myprinter.doprint(1*x) == "x"
    assert myprinter.doprint(MutableDenseMatrix(1, 2, [x, y]), "larry") == "larry[0] = x;\nlarry[1] = y;"
    raises(TypeError, lambda: myprinter.doprint(sin(x), Integer))
    raises(RuntimeError, lambda: myprinter.doprint(MutableDenseMatrix(1, 2, [x, y])))

def test_init_printing():
    x = Symbol("x")
    assert x._repr_latex_() is None
    init_printing()
    assert x._repr_latex_() == '$x$'


def test_unicode():
    x = Symbol("x")
    y = Integer(2)
    assert unicode(x / 2) == "x\n―\n2"
