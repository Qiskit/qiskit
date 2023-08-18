# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The base interface for Opflow's gradient."""

from typing import Union, List, Optional
import numpy as np

from qiskit.circuit import ParameterExpression, ParameterVector
from qiskit.circuit._utils import sort_parameters
from qiskit.utils import optionals as _optionals
from qiskit.utils.deprecation import deprecate_func
from .circuit_gradients.circuit_gradient import CircuitGradient
from ..expectations.pauli_expectation import PauliExpectation
from .gradient_base import GradientBase
from .derivative_base import _coeff_derivative
from ..list_ops.composed_op import ComposedOp
from ..list_ops.list_op import ListOp
from ..list_ops.summed_op import SummedOp
from ..list_ops.tensored_op import TensoredOp
from ..operator_base import OperatorBase
from ..operator_globals import Zero, One
from ..state_fns.circuit_state_fn import CircuitStateFn
from ..exceptions import OpflowError


class Gradient(GradientBase):
    """Deprecated: Convert an operator expression to the first-order gradient."""

    @deprecate_func(
        since="0.24.0",
        additional_msg="For code migration guidelines, visit https://qisk.it/opflow_migration.",
    )
    def __init__(self, grad_method: Union[str, CircuitGradient] = "param_shift", **kwargs):
        super().__init__(grad_method=grad_method, **kwargs)

    def convert(
        self,
        operator: OperatorBase,
        params: Optional[
            Union[ParameterVector, ParameterExpression, List[ParameterExpression]]
        ] = None,
    ) -> OperatorBase:
        r"""
        Args:
            operator: The operator we are taking the gradient of.
            params: The parameters we are taking the gradient with respect to. If not
                explicitly passed, they are inferred from the operator and sorted by name.

        Returns:
            An operator whose evaluation yields the Gradient.

        Raises:
            ValueError: If ``params`` contains a parameter not present in ``operator``.
            ValueError: If ``operator`` is not parameterized.
        """
        if len(operator.parameters) == 0:
            raise ValueError("The operator we are taking the gradient of is not parameterized!")
        if params is None:
            params = sort_parameters(operator.parameters)
        if isinstance(params, (ParameterVector, list)):
            param_grads = [self.convert(operator, param) for param in params]
            absent_params = [
                params[i] for i, grad_ops in enumerate(param_grads) if grad_ops is None
            ]
            if len(absent_params) > 0:
                raise ValueError(
                    "The following parameters do not appear in the provided operator: ",
                    absent_params,
                )
            return ListOp(param_grads)

        param = params
        # Preprocessing
        expec_op = PauliExpectation(group_paulis=False).convert(operator).reduce()
        cleaned_op = self._factor_coeffs_out_of_composed_op(expec_op)
        return self.get_gradient(cleaned_op, param)

    # pylint: disable=too-many-return-statements
    def get_gradient(
        self,
        operator: OperatorBase,
        params: Union[ParameterExpression, ParameterVector, List[ParameterExpression]],
    ) -> OperatorBase:
        """Get the gradient for the given operator w.r.t. the given parameters

        Args:
            operator: Operator w.r.t. which we take the gradient.
            params: Parameters w.r.t. which we compute the gradient.

        Returns:
            Operator which represents the gradient w.r.t. the given params.

        Raises:
            ValueError: If ``params`` contains a parameter not present in ``operator``.
            OpflowError: If the coefficient of the operator could not be reduced to 1.
            OpflowError: If the differentiation of a combo_fn requires JAX but the package is not
                       installed.
            TypeError: If the operator does not include a StateFn given by a quantum circuit
            Exception: Unintended code is reached
            MissingOptionalLibraryError: jax not installed
        """

        def is_coeff_c(coeff, c):
            if isinstance(coeff, ParameterExpression):
                expr = coeff._symbol_expr
                return expr == c
            return coeff == c

        def is_coeff_c_abs(coeff, c):
            if isinstance(coeff, ParameterExpression):
                expr = coeff._symbol_expr
                return np.abs(expr) == c
            return np.abs(coeff) == c

        if isinstance(params, (ParameterVector, list)):
            param_grads = [self.get_gradient(operator, param) for param in params]
            # If get_gradient returns None, then the corresponding parameter was probably not
            # present in the operator. This needs to be looked at more carefully as other things can
            # probably trigger a return of None.
            absent_params = [
                params[i] for i, grad_ops in enumerate(param_grads) if grad_ops is None
            ]
            if len(absent_params) > 0:
                raise ValueError(
                    "The following parameters do not appear in the provided operator: ",
                    absent_params,
                )
            return ListOp(param_grads)

        # By now params is a single parameter
        param = params
        # Handle Product Rules
        if not is_coeff_c(operator._coeff, 1.0) and not is_coeff_c(operator._coeff, 1.0j):
            # Separate the operator from the coefficient
            coeff = operator._coeff
            op = operator / coeff
            if np.iscomplex(coeff):
                from .circuit_gradients.lin_comb import LinComb

                if isinstance(self.grad_method, LinComb):
                    op *= 1j
                    coeff /= 1j

            # Get derivative of the operator (recursively)
            d_op = self.get_gradient(op, param)
            # ..get derivative of the coeff
            d_coeff = _coeff_derivative(coeff, param)

            grad_op = 0
            if d_op != ~Zero @ One and not is_coeff_c(coeff, 0.0):
                grad_op += coeff * d_op
            if op != ~Zero @ One and not is_coeff_c(d_coeff, 0.0):
                grad_op += d_coeff * op
            if grad_op == 0:
                grad_op = ~Zero @ One
            return grad_op

        # Base Case, you've hit a ComposedOp!
        # Prior to execution, the composite operator was standardized and coefficients were
        # collected. Any operator measurements were converted to Pauli-Z measurements and rotation
        # circuits were applied. Additionally, all coefficients within ComposedOps were collected
        # and moved out front.
        if isinstance(operator, ComposedOp):

            # Gradient of an expectation value
            if not is_coeff_c_abs(operator._coeff, 1.0):
                raise OpflowError(
                    "Operator pre-processing failed. Coefficients were not properly "
                    "collected inside the ComposedOp."
                )

            # Do some checks to make sure operator is sensible
            # TODO add compatibility with sum of circuit state fns
            if not isinstance(operator[-1], CircuitStateFn):
                raise TypeError(
                    "The gradient framework is compatible with states that are given as "
                    "CircuitStateFn"
                )

            return self.grad_method.convert(operator, param)

        elif isinstance(operator, CircuitStateFn):
            # Gradient of an a state's sampling probabilities
            if not is_coeff_c(operator._coeff, 1.0):
                raise OpflowError(
                    "Operator pre-processing failed. Coefficients were not properly "
                    "collected inside the ComposedOp."
                )
            return self.grad_method.convert(operator, param)

        # Handle the chain rule
        elif isinstance(operator, ListOp):
            grad_ops = [self.get_gradient(op, param) for op in operator.oplist]

            # pylint: disable=comparison-with-callable
            if operator.combo_fn == ListOp.default_combo_fn:  # If using default
                return ListOp(oplist=grad_ops)
            elif isinstance(operator, SummedOp):
                return SummedOp(oplist=[grad for grad in grad_ops if grad != ~Zero @ One]).reduce()
            elif isinstance(operator, TensoredOp):
                return TensoredOp(oplist=grad_ops)

            if operator.grad_combo_fn:
                grad_combo_fn = operator.grad_combo_fn
            else:
                _optionals.HAS_JAX.require_now("automatic differentiation")
                from jax import jit, grad

                grad_combo_fn = jit(grad(operator.combo_fn, holomorphic=True))

            def chain_rule_combo_fn(x):
                result = np.dot(x[1], x[0])
                if isinstance(result, np.ndarray):
                    result = list(result)
                return result

            return ListOp(
                [ListOp(operator.oplist, combo_fn=grad_combo_fn), ListOp(grad_ops)],
                combo_fn=chain_rule_combo_fn,
            )
