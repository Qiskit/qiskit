# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from qiskit.utils import optionals

if optionals.HAS_SYMENGINE:
    import symengine as sym
else:
    import sympy as sym


def _normalized_gaussian(t, center, zeroed_width, sigma):
    gauss = sym.exp(-((t - center) / sigma) ** 2 / 2)

    t_edge = zeroed_width / 2
    offset = sym.exp(-(t_edge / sigma) ** 2 / 2)

    return (gauss - offset) / (1 - offset)


def symbolic_gaussian():
    t, duration, amp, sigma = sym.symbols("t, duration, amp, sigma")
    center = duration / 2

    return amp * _normalized_gaussian(t, center, duration + 2, sigma)


def symbolic_gaussian_square():
    t, duration, amp, sigma, width = sym.symbols("t, duration, amp, sigma, width")
    center = duration / 2

    sq_t0 = center - width / 2
    sq_t1 = center + width / 2
    gaussian_zeroed_width = duration + 2 - width

    gaussian_ledge = _normalized_gaussian(t, sq_t0, gaussian_zeroed_width, sigma)
    gaussian_redge = _normalized_gaussian(t, sq_t1, gaussian_zeroed_width, sigma)

    return amp * sym.Piecewise((gaussian_ledge, t <= sq_t0), (gaussian_redge, t >= sq_t1), (1, True))


def symbolic_drag():
    t, duration, amp, sigma, beta = sym.symbols("t, duration, amp, sigma, beta")
    center = duration / 2

    gauss = amp * _normalized_gaussian(t, center, duration + 2, sigma)
    deriv = - (t - center) / sigma * gauss

    return gauss + 1j * beta * deriv


def symbolic_constant():
    t, duration, amp = sym.symbols("t, duration, amp")

    # Note this is implemented using Piecewise instead of just returning amp
    # directly because otherwise the expression has no t dependence and sympy's
    # lambdify will produce a function f that for an array t returns amp
    # instead of amp * np.ones(t.shape). This does not work well with
    # ParametricPulse.get_waveform().
    #
    # See: https://github.com/sympy/sympy/issues/5642
    return amp * sym.Piecewise((1, 0<= t <= duration), (0, True))
