# -*- coding: utf-8 -*-

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

"""
A collection of universal functions applied to Parameters.
"""

from .parameterexpression import ParameterExpression


def _call(ufunc, args) -> ParameterExpression:
    """Return a new ParameterExpression with ufunc applied to the old one."""
    if not isinstance(args, ParameterExpression):
        raise TypeError('Input should be a ParameterExpression')
    if args != args.conjugate():
        raise ValueError('Input should be a Real ParameterExpression.'
                         'In case you want to apply complex exponential, see cexp function.')

    return ParameterExpression(
        args._parameter_symbols,
        ufunc(args._symbol_expr)
    )


def sin(x):
    """Sine of a ParameterExpression"""
    from sympy import sin as _sin
    return _call(_sin, x)


def cos(x):
    """Cosine of a ParameterExpression"""
    from sympy import cos as _cos
    return _call(_cos, x)


def tan(x):
    """Tangent of a ParameterExpression"""
    from sympy import tan as _tan
    return _call(_tan, x)


def asin(x):
    """Arcsin of a ParameterExpression"""
    from sympy import asin as _asin
    return _call(_asin, x)


def acos(x):
    """Arccos of a ParameterExpression"""
    from sympy import acos as _acos
    return _call(_acos, x)


def atan(x):
    """Arctan of a ParameterExpression"""
    from sympy import atan as _atan
    return _call(_atan, x)


# proper exponential
def exp(x):
    """Exponential of a ParameterExpression"""
    from sympy import exp as _exp
    return _call(_exp, x)


# complex exponential
def cexp(x):
    """Complex exponential of a ParameterExpression"""
    from sympy import I
    from sympy import exp as _exp
    return _call(
        lambda x: _exp(I * x),
        x)


def log(x):
    """Logarithm of a ParameterExpression"""
    from sympy import log as _log
    return _call(_log, x)
