from symengine.test_utilities import raises

from symengine.lib.symengine_wrapper import (isprime, nextprime, gcd,
    lcm, gcd_ext, mod, quotient, quotient_mod, mod_inverse, crt, fibonacci,
    fibonacci2, lucas, lucas2, binomial, factorial, divides, factor,
    factor_lehman_method, factor_pollard_pm1_method, factor_pollard_rho_method,
    prime_factors, prime_factor_multiplicities, Sieve, Sieve_iterator,
    bernoulli, primitive_root, primitive_root_list, totient, carmichael,
    multiplicative_order, legendre, jacobi, kronecker, nthroot_mod,
    nthroot_mod_list, powermod, powermod_list, Integer, sqrt_mod)


def test_probab_prime_p():
    s = set(Sieve.generate_primes(1000))
    for n in range(1001):
        assert (n in s) == isprime(n)


def test_nextprime():
    assert nextprime(-3) == 2
    assert nextprime(5) == 7
    assert nextprime(9) == 11


def test_gcd():
    assert gcd(10, 14) == 2
    assert gcd(-5, 3) == 1


def test_lcm():
    assert lcm(10, 14) == 70
    assert lcm(-5, 3) == 15


def test_gcd_ext():
    (q, r, p) = gcd_ext(6, 9)
    assert p == q * 6 + r * 9
    (q, r, p) = gcd_ext(-15, 10)
    assert p == q * -15 + r * 10
    (q, r, p) = gcd_ext(2, 3)
    assert p == q * 2 + r * 3
    assert p == 1
    (q, r, p) = gcd_ext(10, 12)
    assert p == q * 10 + r * 12
    assert p == 2
    (q, r, p) = gcd_ext(100, 2004)
    assert p == q * 100 + r * 2004
    assert p == 4


def test_mod():
    assert mod(13, 5) == 3
    assert mod(-4, 7) == 3


def test_mod_error():
    raises(ZeroDivisionError, lambda: mod(2, 0))


def test_quotient():
    assert quotient(13, 5) == 2
    assert quotient(-4, 7) == -1


def test_quotient_error():
    raises(ZeroDivisionError, lambda: quotient(1, 0))


def test_quotient_mod():
    assert quotient_mod(13, 5) == (2, 3)
    assert quotient_mod(-4, 7) == (-1, 3)


def test_quotient_mod_error():
    raises(ZeroDivisionError, lambda: quotient_mod(1, 0))


def test_mod_inverse():
    mod_inverse(2, 7) == 4
    mod_inverse(0, 3) is None
    mod_inverse(4, 6) is None


def test_crt():
    assert crt([0, 1, 2, 4], [2, 3, 4, 5]) == 34
    assert crt([3, 5], [6, 21]) is None


def test_fibonacci():
    assert fibonacci(0) == 0
    assert fibonacci(5) == 5


def test_fibonacci_error():
    raises(NotImplementedError, lambda: fibonacci(-3))


def test_fibonacci2():
    assert fibonacci2(0) == [0, 1]
    assert fibonacci2(5) == [5, 3]


def test_fibonacci2_error():
    raises(NotImplementedError, lambda: fibonacci2(-1))


def test_lucas():
    assert lucas(2) == 3
    assert lucas(3) == 4


def test_lucas_error():
    raises(NotImplementedError, lambda: lucas(-1))


def test_lucas2():
    assert lucas2(3) == [4, 3]
    assert lucas2(5) == [11, 7]


def test_lucas2_error():
    raises(NotImplementedError, lambda: lucas2(-1))


def test_binomial():
    assert binomial(5, 2) == 10
    assert binomial(5, 7) == 0
    assert binomial(-5, 2) == 15


def test_binomial_error():
    raises(ArithmeticError, lambda: binomial(5, -1))


def test_factorial():
    assert factorial(5) == 120
    assert factorial(0) == 1


def test_factorial_error():
    raises(ArithmeticError, lambda: factorial(-1))


def test_divides():
    assert divides(5, 2) is False
    assert divides(10, 5) is True
    assert divides(0, 0) is True
    assert divides(5, 0) is False


def test_factor():
    f = factor(102)
    assert f is not None and divides(102, f)
    assert factor(101) is None


def test_prime_factors():
    assert prime_factors(100) == [2, 2, 5, 5]
    assert prime_factors(1) == []


def test_prime_factor_multiplicities():
    assert prime_factor_multiplicities(90) == \
           {Integer(2): 1, Integer(3): 2, Integer(5): 1}
    assert prime_factor_multiplicities(1) == {}


def test_sieve():
    assert Sieve.generate_primes(50) == [2, 3, 5, 7, 11, 13, 17,
                                         19, 23, 29, 31, 37, 41, 43, 47]
    assert len(Sieve.generate_primes(1009)) == 169


def test_sieve_iterator():
    it = Sieve_iterator(101)
    assert len([i for i in it]) == 26


def test_primitive_root():
    assert primitive_root(27) in [2, 5, 11, 14, 20, 23]
    assert primitive_root(15) is None


def test_primitive_root_list():
    assert primitive_root_list(54) == [5, 11, 23, 29, 41, 47]
    assert primitive_root_list(12) == []


def test_totient():
    assert totient(1) == 1
    assert totient(-15) == 8


def test_carmichael():
    assert carmichael(8) == 2
    assert carmichael(-21) == 6


def test_multiplicative_order():
    assert multiplicative_order(2, 21) == 6
    assert multiplicative_order(5, 10) is None


def test_legendre():
    assert legendre(-1, 5) == 1
    assert legendre(0, 5) == 0
    assert legendre(2, 5) == -1


def test_jacobi():
    assert legendre(-1, 77) == 1
    assert legendre(14, 15) == -1


def test_kronecker():
    assert kronecker(9, 2) == 1
    assert kronecker(-5, -1) == -1


def test_nthroot_mod():
    assert nthroot_mod(12, 5, 77) in [3, 31, 38, 45, 59]
    assert nthroot_mod(3, 2, 5) is None


def test_sqrt_mod():
    assert sqrt_mod(3, 13) == 9
    assert sqrt_mod(6, 23) == 12
    assert sqrt_mod(345, 690) == 345
    assert sqrt_mod(9, 27, True) == [3, 6, 12, 15, 21, 24]
    assert sqrt_mod(9, 81, True) == [3, 24, 30, 51, 57, 78]
    assert sqrt_mod(9, 3**5, True) == [3, 78, 84, 159, 165, 240]
    assert sqrt_mod(81, 3**4, True) == [0, 9, 18, 27, 36, 45, 54, 63, 72]
    assert sqrt_mod(81, 3**5, True) == [9, 18, 36, 45, 63, 72, 90, 99, 117,\
            126, 144, 153, 171, 180, 198, 207, 225, 234]
    assert sqrt_mod(81, 3**6, True) == [9, 72, 90, 153, 171, 234, 252, 315,\
            333, 396, 414, 477, 495, 558, 576, 639, 657, 720]
    assert sqrt_mod(81, 3**7, True) == [9, 234, 252, 477, 495, 720, 738, 963,\
            981, 1206, 1224, 1449, 1467, 1692, 1710, 1935, 1953, 2178]


def test_nthroot_mod_list():
    assert nthroot_mod_list(-4, 4, 65) == [4, 6, 7, 9, 17, 19, 22, 32,
                                           33, 43, 46, 48, 56, 58, 59, 61]
    assert nthroot_mod_list(2, 3, 7) == []


def test_powermod():
    assert powermod(3, 8, 7) == 2
    assert powermod(3, Integer(11)/2, 13) in [3, 10]


def test_powermod_list():
    assert powermod_list(15, Integer(1)/6, 21) == [3, 6, 9, 12, 15, 18]
    assert powermod_list(2, Integer(5)/2, 11) == []
