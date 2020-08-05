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
    from sympy import sin as _sin
    return _call(_sin, x)


def cos(x):
    from sympy import cos as _cos
    return _call(_cos, x)


def tan(x):
    from sympy import tan as _tan
    return _call(_tan, x)


def asin(x):
    from sympy import asin as _asin
    return _call(_asin, x)


def acos(x):
    from sympy import acos as _acos
    return _call(_acos, x)


def atan(x):
    from sympy import atan as _atan
    return _call(_atan, x)


# proper exponential
def exp(x):
    from sympy import exp as _exp
    return _call(_exp, x)


# complex exponential
def cexp(x):
    from sympy import I
    from sympy import exp as _exp
    return _call(
        lambda x: _exp(I * x),
        x)


def log(x):
    from sympy import log as _log
    return _call(_log, x)
