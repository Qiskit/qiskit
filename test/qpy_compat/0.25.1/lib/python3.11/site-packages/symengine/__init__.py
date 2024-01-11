import os
import sys

if sys.version_info >= (3, 8, 0) and sys.platform == 'win32' \
       and 'SYMENGINE_PY_ADD_PATH_TO_SEARCH_DIRS' in os.environ:
    for directory in os.environ['PATH'].split(';'):
        if os.path.isdir(directory):
            os.add_dll_directory(directory)

del os, sys

import symengine.lib.symengine_wrapper as wrapper

from .lib.symengine_wrapper import (
    have_mpfr, have_mpc, have_flint, have_piranha, have_llvm, have_llvm_long_double,
    I, E, pi, oo, zoo, nan, Symbol, Dummy, S, sympify, SympifyError,
    Integer, Rational, Float, Number, RealNumber, RealDouble, ComplexDouble,
    add, Add, Mul, Pow, function_symbol,
    Max, Min, DenseMatrix, Matrix,
    ImmutableMatrix, ImmutableDenseMatrix, MutableDenseMatrix,
    MatrixBase, Basic, DictBasic, symarray, series, diff, zeros,
    eye, diag, ones, Derivative, Subs, expand, has_symbol,
    UndefFunction, Function, UnevaluatedExpr, latex,
    have_numpy, true, false, Equality, Unequality, GreaterThan,
    LessThan, StrictGreaterThan, StrictLessThan, Eq, Ne, Ge, Le,
    Gt, Lt, And, Or, Not, Nand, Nor, Xor, Xnor, perfect_power, integer_nthroot,
    isprime, sqrt_mod, Expr, cse, count_ops, ccode, Piecewise, Contains, Interval, FiniteSet,
    linsolve,
    FunctionSymbol as AppliedUndef,
    golden_ratio as GoldenRatio,
    catalan as Catalan,
    eulergamma as EulerGamma,
    unicode
)
from .utilities import var, symbols
from .functions import *
from .printing import init_printing


EmptySet = wrapper.S.EmptySet
UniversalSet = wrapper.S.UniversalSet
Reals = wrapper.S.Reals
Integers = wrapper.S.Integers
Rationals = wrapper.S.Rationals


if have_mpfr:
    from .lib.symengine_wrapper import RealMPFR

if have_mpc:
    from .lib.symengine_wrapper import ComplexMPC

if have_numpy:
    from .lib.symengine_wrapper import (Lambdify, LambdifyCSE)

    def lambdify(args, exprs, **kwargs):
        return Lambdify(args, *exprs, **kwargs)


__version__ = "0.9.2"


# To not expose internals
del lib.symengine_wrapper
del lib
del wrapper


def test():
    import pytest
    import os
    return not pytest.cmdline.main(
        [os.path.dirname(os.path.abspath(__file__))])
