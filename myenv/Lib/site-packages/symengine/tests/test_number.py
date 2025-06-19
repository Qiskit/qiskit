from symengine.test_utilities import raises

from symengine import Integer, I, S, Symbol, pi, Rational
from symengine.lib.symengine_wrapper import (perfect_power, is_square, integer_nthroot)


def test_integer():
    i = Integer(5)
    assert str(i) == "5"
    assert int(i) == 5
    assert float(i) == 5.0
    assert complex(i) == 5.0 + 0j
    assert i.real == i
    assert i.imag == S.Zero


def test_integer_long():
    py_int = 123434444444444444444
    i = Integer(py_int)
    assert str(i) == str(py_int)
    assert int(i) == py_int


def test_integer_string():
    assert Integer("133") == 133


def test_rational():
    i = Integer(5)/10
    assert str(i) == "1/2"
    assert int(i) == 0
    assert float(i) == 0.5
    assert complex(i) == 0.5 + 0j
    assert i.real == i
    assert i.imag == S.Zero


def test_complex():
    i = Integer(5)/10 + I
    assert str(i) == "1/2 + I"
    assert complex(i) == 0.5 + 1j
    assert i.real == Integer(1)/2
    assert i.imag == 1

    i = 0.5 + I
    assert str(i) == "0.5 + 1.0*I"
    assert complex(i) == 0.5 + 1j
    assert i.real == 0.5
    assert i.imag == 1.0


def test_smallfloat_valid():
    i = Integer(7.5)
    assert str(i) == "7"


def test_bigfloat_valid():
    i = Integer(13333333333333334.5)
    assert str(i) == "13333333333333334"


def test_is_conditions():
    i = Integer(-123)
    assert not i.is_zero
    assert not i.is_positive
    assert i.is_negative
    assert i.is_nonzero
    assert i.is_nonpositive
    assert not i.is_nonnegative
    assert not i.is_complex

    i = Integer(123)
    assert not i.is_zero
    assert i.is_positive
    assert not i.is_negative
    assert i.is_nonzero
    assert not i.is_nonpositive
    assert i.is_nonnegative
    assert not i.is_complex

    i = Integer(0)
    assert i.is_zero
    assert not i.is_positive
    assert not i.is_negative
    assert not i.is_nonzero
    assert i.is_nonpositive
    assert i.is_nonnegative
    assert not i.is_complex

    i = Integer(1) + I
    assert not i.is_zero
    assert not i.is_positive
    assert not i.is_negative
    assert not i.is_nonzero
    assert not i.is_nonpositive
    assert not i.is_nonnegative
    assert i.is_complex

    assert pi.is_number


def test_perfect_power():
    assert perfect_power(1) == True
    assert perfect_power(7) == False
    assert perfect_power(8) == True
    assert perfect_power(9) == True
    assert perfect_power(10) == False
    assert perfect_power(1024) == True
    assert perfect_power(1025) == False
    assert perfect_power(6**7) == True
    assert perfect_power(-27) == True
    assert perfect_power(-64) == True
    assert perfect_power(-32) == True


def test_perfect_square():
    assert is_square(7) == False
    assert is_square(8) == False
    assert is_square(9) == True
    assert is_square(10) == False
    assert perfect_power(49) == True
    assert perfect_power(50) == False


def test_integer_nthroot():
    assert integer_nthroot(1, 2) == (1, True)
    assert integer_nthroot(1, 5) == (1, True)
    assert integer_nthroot(2, 1) == (2, True)
    assert integer_nthroot(2, 2) == (1, False)
    assert integer_nthroot(2, 5) == (1, False)
    assert integer_nthroot(4, 2) == (2, True)
    assert integer_nthroot(123**25, 25) == (123, True)
    assert integer_nthroot(123**25 + 1, 25) == (123, False)
    assert integer_nthroot(123**25 - 1, 25) == (122, False)
    assert integer_nthroot(1, 1) == (1, True)
    assert integer_nthroot(0, 1) == (0, True)
    assert integer_nthroot(0, 3) == (0, True)
    assert integer_nthroot(10000, 1) == (10000, True)
    assert integer_nthroot(4, 2) == (2, True)
    assert integer_nthroot(16, 2) == (4, True)
    assert integer_nthroot(26, 2) == (5, False)
    assert integer_nthroot(1234567**7, 7) == (1234567, True)
    assert integer_nthroot(1234567**7 + 1, 7) == (1234567, False)
    assert integer_nthroot(1234567**7 - 1, 7) == (1234566, False)
    b = 25**1000
    assert integer_nthroot(b, 1000) == (25, True)
    assert integer_nthroot(b + 1, 1000) == (25, False)
    assert integer_nthroot(b - 1, 1000) == (24, False)
    c = 10**400
    c2 = c**2
    assert integer_nthroot(c2, 2) == (c, True)
    assert integer_nthroot(c2 + 1, 2) == (c, False)
    assert integer_nthroot(c2 - 1, 2) == (c - 1, False)


def test_is_zero():
    assert Symbol('x').is_zero is None


def test_is_positive():
    assert Rational(1, 2).is_positive
    assert not Rational(-2, 3).is_positive
    assert Symbol('x').is_positive is None


def test_is_negative():
    assert not Rational(1, 2).is_negative
    assert Rational(-2, 3).is_negative
    assert Symbol('x').is_negative is None


def test_is_nonpositive():
    assert not Rational(1, 2).is_nonpositive
    assert Rational(-2, 3).is_nonpositive
    assert Symbol('x').is_nonpositive is None


def test_is_nonnegative():
    assert Rational(1, 2).is_nonnegative
    assert not Rational(-2, 3).is_nonnegative
    assert Symbol('x').is_nonnegative is None


def test_is_real():
    assert Rational(1, 2).is_real
    assert Symbol('x').is_real is None
