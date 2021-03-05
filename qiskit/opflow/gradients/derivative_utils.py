from plum import dispatch
from qiskit.circuit import ParameterExpression

# An opflow expression representing 0.0 (that evaluates to zero)
#ZERO_EXPR = ~Zero @ One
ZERO_EXPR = 0.0

@dispatch
def is_coeff_c(coeff: float, c: {float, int}):
    return coeff == c

@dispatch
def is_coeff_c(coeff: ParameterExpression, c: {float, int}):
    return coeff._symbol_expr == c
