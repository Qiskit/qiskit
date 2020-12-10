# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" ae utilities """

import logging
import numpy as np

logger = logging.getLogger(__name__)

# pylint: disable=invalid-name


def bisect_max(f, a, b, steps=50, minwidth=1e-12, retval=False):
    """
    Find the maximum of the real-valued function f in the interval [a, b]
    using bisection

    Args:
        f (callable): the function to find the maximum of
        a (float): the lower limit of the interval
        b (float): the upper limit of the interval
        steps (int): the maximum number of steps in the bisection
        minwidth (float): if the current interval is smaller than minwidth stop
            the search
        retval (bool): return value

    Returns:
        float: The maximum of f in [a,b] according to this algorithm.
    """
    it = 0
    m = (a + b) / 2
    fm = 0
    while it < steps and b - a > minwidth:
        l, r = (a + m) / 2, (m + b) / 2
        fl, fm, fr = f(l), f(m), f(r)

        # fl is the maximum
        if fl > fm and fl > fr:
            b = m
            m = l
        # fr is the maximum
        elif fr > fm and fr > fl:
            a = m
            m = r
        # fm is the maximum
        else:
            a = l
            b = r

        it += 1

    if it == steps:
        logger.warning("-- Warning, bisect_max didn't converge after %s steps", steps)

    if retval:
        return m, fm

    return m


def circ_dist(x, p):
    """
    Circumferential distance and derivative function,
            d(x, p) = min_{z in [-1, 0, 1]} (|z + p - x|)

    Args:
        x (float): first angle
        p (float): second angle

    Returns:
        float: d(x, p)
    """
    t = p - x
    # Since x and p \in [0,1] it suffices to check not all integers
    # but only -1, 0 and 1
    z = np.array([-1, 0, 1])

    if hasattr(t, "__len__"):
        d = np.empty_like(t)
        for idx, ti in enumerate(t):
            d[idx] = np.min(np.abs(z + ti))
        return d

    return np.min(np.abs(z + t))


def derivative_circ_dist(x, p):
    """
    Derivative of circumferential distance and derivative function, w.r.t. p
            d/dp d(x, p) = d/dp min_{z in [-1, 0, 1]} (|z + p - x|)

    Args:
        x (float): first angle
        p (float): second angle

    Returns:
        float: d/dp d(x, p)
    """
    # pylint: disable=chained-comparison,misplaced-comparison-constant
    t = p - x
    if t < -0.5 or (0 < t and t < 0.5):
        return -1
    if t > 0.5 or (-0.5 < t and t < 0):
        return 1
    return 0


def omega(a):
    """
    Transform from the value a to the corresponding angle omega.

    Args:
        a (float): A value in [0,1]

    Returns:
        float: arcsin(sqrt(a)) / pi
    """
    return np.arcsin(np.sqrt(a)) / np.pi


def derivative_omega(a):
    """
    Compute the derivative of omega.
    """
    return 1 / (2 * np.pi * np.sqrt((1 - a) * a))


def alpha(x, p):
    """
    Helper function for `pdf_a`, alpha = pi * d(omega(x), omega(p)),
    """
    return np.pi * circ_dist(omega(x), omega(p))


def derivative_alpha(x, p):
    """
    Compute the derivative of alpha.
    """
    return np.pi * derivative_circ_dist(omega(x), omega(p)) * derivative_omega(p)


def beta(x, p):
    """
    Helper function for `pdf_a`, beta = pi * d(1 - omega(x), omega(p)),
    """
    return np.pi * circ_dist(1 - omega(x), omega(p))


def derivative_beta(x, p):
    """
    Compute the derivative of beta.
    """
    return np.pi * derivative_circ_dist(1 - omega(x), omega(p)) * derivative_omega(p)


def pdf_a_single_angle(x, p, m, pi_delta):
    """
    Helper function for `pdf_a`.
    """
    M = 2**m

    d = pi_delta(x, p)
    res = np.sin(M * d)**2 / (M * np.sin(d))**2 if d != 0 else 1

    return res


def pdf_a(x, p, m):
    """
    Return the PDF of a, i.e. the probability of getting the estimate x
    (in [0, 1]) if p (in [0, 1]) is the true value, given that we use m qubits.

    Args:
        x (float): the grid point
        p (float): the true value
        m (float): the number of evaluation qubits

    Returns:
        float: PDF(x|p)
    """
    # We'll use list comprehension, so the input should be a list
    scalar = False
    if not hasattr(x, "__len__"):
        scalar = True
        x = np.asarray([x])

    # Compute the probabilities: Add up both angles that produce the given
    # value, except for the angles 0 and 0.5, which map to the unique a-values,
    # 0 and 1, respectively
    pr = np.array([pdf_a_single_angle(xi, p, m, alpha) + pdf_a_single_angle(xi, p, m, beta)
                   if (xi not in [0, 1]) else pdf_a_single_angle(xi, p, m, alpha)
                   for xi in x
                   ]).flatten()

    # If is was a scalar return scalar otherwise the array
    return pr[0] if scalar else pr


def derivative_log_pdf_a(x, p, m):
    """
    Return the derivative of the logarithm of the PDF of a.

    Args:
        x (float): the grid point
        p (float): the true value
        m (float): the number of evaluation qubits

    Returns:
        float: d/dp log(PDF(x|p))
    """
    M = 2**m

    if x not in [0, 1]:
        num_p1 = 0
        for A, dA, B, dB in zip([alpha, beta],
                                [derivative_alpha, derivative_beta],
                                [beta, alpha], [derivative_beta, derivative_alpha]):
            num_p1 += 2 * M * np.sin(M * A(x, p)) * np.cos(M * A(x, p)) \
                * dA(x, p) * np.sin(B(x, p))**2 \
                + 2 * np.sin(M * A(x, p))**2 * np.sin(B(x, p)) * np.cos(B(x, p)) * dB(x, p)

        den_p1 = np.sin(M * alpha(x, p))**2 * np.sin(beta(x, p))**2 + \
            np.sin(M * beta(x, p))**2 * np.sin(alpha(x, p))**2

        num_p2 = 0
        for A, dA, B in zip([alpha, beta], [derivative_alpha, derivative_beta], [beta, alpha]):
            num_p2 += 2 * np.cos(A(x, p)) * dA(x, p) * np.sin(B(x, p))

        den_p2 = np.sin(alpha(x, p)) * np.sin(beta(x, p))

        return num_p1 / den_p1 - num_p2 / den_p2

    return 2 * derivative_alpha(x, p) * (M / np.tan(M * alpha(x, p)) - 1 / np.tan(alpha(x, p)))
