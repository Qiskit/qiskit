from multipledispatch import dispatch
from qiskit.circuit import ParameterExpression

# An opflow expression representing 0.0 (that evaluates to zero)
#ZERO_EXPR = ~Zero @ One
ZERO_EXPR = 0.0

@dispatch(float, (float, int))
def is_coeff_c(coeff, c):
    return coeff == c

@dispatch(ParameterExpression, (float, int))
def is_coeff_c(coeff, c):
    return coeff._symbol_expr == c
#    return coeff == c

