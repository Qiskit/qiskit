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

"""The module to compute Hessians."""

from typing import Union, List, Tuple

# This requires the master branch of plum-dispatch
import plum

import numpy as np
from qiskit.exceptions import MissingOptionalLibraryError
from qiskit.circuit import ParameterVector, ParameterExpression, Parameter
from ..state_fns.circuit_state_fn import CircuitStateFn
from ..state_fns.state_fn import StateFn
from ..expectations.pauli_expectation import PauliExpectation
from ..list_ops.list_op import ListOp
from ..list_ops.composed_op import ComposedOp
from ..list_ops.summed_op import SummedOp
from ..list_ops.tensored_op import TensoredOp
from ..operator_base import OperatorBase
from .gradient import Gradient
from .hessian_base import HessianBase
from ..exceptions import OpflowError
from ...utils.arithmetic import triu_to_dense

try:
    from jax import grad, jit
    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False

from .derivative_utils import ZERO_EXPR
from .derivative_utils import is_coeff_c

class Hessian(HessianBase):
    """Compute the Hessian of an expected value."""

    dispatch = plum.Dispatcher(in_class=plum.Self)

    # pylint: disable=signature-differs
    def convert(self,
                operator: OperatorBase,
                params: Union[Tuple[ParameterExpression, ParameterExpression],
                              List[Tuple[ParameterExpression, ParameterExpression]],
                              List[ParameterExpression], ParameterVector]
                ) -> OperatorBase:
        """
        Args:
            operator: The operator for which we compute the Hessian
            params: The parameters we are computing the Hessian with respect to
                    Either give directly the tuples/list of tuples for which the second order
                    derivative is to be computed or give a list of parameters to build the
                    full Hessian for those parameters.

        Returns:
            OperatorBase: An operator whose evaluation yields the Hessian
        """

        expec_op = PauliExpectation(group_paulis=False).convert(operator).reduce()
        cleaned_op = self._factor_coeffs_out_of_composed_op(expec_op)

        return self.get_hessian(cleaned_op, params)

    @dispatch
    def get_hessian(self, operator: object, params: ParameterExpression, precedence=1):
        return Gradient(grad_method=self._hess_method).get_gradient(operator, params)

    @dispatch
    def get_hessian(self, operator: object, params: plum.List(tuple)):
        # Compute the Hessian entries corresponding to these pairs of parameters.
        return ListOp([self.get_hessian(operator, param_pair) for param_pair in params])

    @dispatch
    def get_hessian(self, operator: object,
                    params: {ParameterVector, plum.List(ParameterExpression)}):
        """Get the Hessian for the given operator w.r.t. the given parameters

        Args:
            operator: Operator w.r.t. which we take the Hessian.
            params: Parameters w.r.t. which we compute the Hessian.

        Returns:
            Operator which represents the gradient w.r.t. the given params.

        Raises:
            ValueError: If ``params`` contains a parameter not present in ``operator``.
            OpflowError: If the coefficient of the operator could not be reduced to 1.
            OpflowError: If the differentiation of a combo_fn
                         requires JAX but the package is not installed.
            TypeError: If the operator does not include a StateFn given by a quantum circuit
            TypeError: If the parameters were given in an unsupported format.
            Exception: Unintended code is reached
            MissingOptionalLibraryError: jax not installed
        """
        return ListOp([ListOp([
            self.get_hessian(operator, (p_i, p_j))
            for i, p_i in enumerate(params[j:], j)])
                       for j, p_j in enumerate(params)],
                      combo_fn=triu_to_dense)

    @dispatch
    def get_hessian(self, operator: object, params: object):
        if is_coeff_c(operator._coeff, 1.0):
            return self.get_hessian_coeff_one(operator, params)
        # Separate the operator from the coefficient
        coeff = operator._coeff
        op = operator / coeff
        # Get derivative of the operator (recursively)
        # By this point, it's only one parameter tuple
        p_0 = params[0]
        p_1 = params[1]

        d0_op = self.get_hessian_coeff_one(op, p_0)
        d1_op = self.get_hessian_coeff_one(op, p_1)
        # ..get derivative of the coeff
        d0_coeff = self.parameter_expression_grad(coeff, p_0)
        d1_coeff = self.parameter_expression_grad(coeff, p_1)

        dd_op = self.get_hessian_coeff_one(op, params)
        dd_coeff = self.parameter_expression_grad(d0_coeff, p_1)

        grad_op = 0
        # Avoid creating operators that will evaluate to zero
        if dd_op != ZERO_EXPR and not is_coeff_c(coeff, 0):
            grad_op += coeff * dd_op
        if d0_op != ZERO_EXPR and not is_coeff_c(d1_coeff, 0):
            grad_op += d1_coeff * d0_op
        if d1_op != ZERO_EXPR and not is_coeff_c(d0_coeff, 0):
            grad_op += d0_coeff * d1_op
        if not is_coeff_c(dd_coeff, 0):
            grad_op += dd_coeff * op

        if grad_op == 0:
            return ZERO_EXPR

        return grad_op

    @dispatch.multi((ComposedOp, tuple), (ComposedOp, Parameter))
    def get_hessian_coeff_one(self, operator: ComposedOp, params: {Parameter, tuple}):
        # Prior to execution, the composite operator was standardized and coefficients were
        # collected. Any operator measurements were converted to Pauli-Z measurements and rotation
        # circuits were applied. Additionally, all coefficients within ComposedOps were collected
        # and moved out front.
        # Do some checks to make sure operator is sensible
        # TODO enable compatibility with sum of CircuitStateFn operators
        if not isinstance(operator[-1], CircuitStateFn):
            raise TypeError(
                'The gradient framework is compatible with states that are given as '
                'CircuitStateFn')

        return self.hess_method.convert(operator, params)

    @dispatch
    def get_hessian_coeff_one(self, operator: {SummedOp, TensoredOp}, params: tuple):
        dd_ops = [self.get_hessian(op, params) for op in operator.oplist]
        return type(operator)(oplist=dd_ops)

    @dispatch
    def get_hessian_coeff_one(self, operator: StateFn, params: tuple):
        if not operator.is_measurement:
            return self.hess_method.convert(operator, params)
        else:
            raise TypeError('The computation of Hessians is only supported for Operators which '
                            'represent expectation values or quantum states.')

    # This is the recursive case where the chain rule is handled
    @dispatch
    def get_hessian_coeff_one(self, operator: ListOp, params: tuple):
        # These operators correspond to (d_op/d θ0,θ1) for op in operator.oplist
        # and params = (θ0,θ1)
        dd_ops = [self.get_hessian(op, params) for op in operator.oplist]

        # Note that this check to see if the ListOp has a default combo_fn
        # will fail if the user manually specifies the default combo_fn.
        # I.e operator = ListOp([...], combo_fn=lambda x:x) will not pass this check and
        # later on jax will try to differentiate it and fail.
        # An alternative is to check the byte code of the operator's combo_fn against the
        # default one.
        # This will work but look very ugly and may have other downsides I'm not aware of
        if operator.combo_fn == ListOp([]).combo_fn:
            return ListOp(oplist=dd_ops)

        # These operators correspond to (d g_i/d θ0)•(d g_i/d θ1) for op in operator.oplist
        # and params = (θ0,θ1)
        d1d0_ops = ListOp([ListOp([Gradient(grad_method=self._hess_method).convert(op, param)
                                   for param in params], combo_fn=np.prod) for
                           op in operator.oplist])

        if operator.grad_combo_fn:
            first_partial_combo_fn = operator.grad_combo_fn
            if _HAS_JAX:
                second_partial_combo_fn = jit(grad(lambda x: first_partial_combo_fn(x)[0],
                                                   holomorphic=True))
            else:
                raise MissingOptionalLibraryError(
                    libname='jax',
                    name='get_hessian',
                    msg='This automatic differentiation function is based on JAX. Please '
                    'install jax and use `import jax.numpy as jnp` instead of '
                    '`import numpy as np` when defining a combo_fn.')
        else:
            if _HAS_JAX:
                first_partial_combo_fn = jit(grad(operator.combo_fn, holomorphic=True))
                second_partial_combo_fn = jit(grad(lambda x: first_partial_combo_fn(x)[0],
                                                   holomorphic=True))
            else:
                raise MissingOptionalLibraryError(
                    libname='jax',
                    name='get_hessian',
                    msg='This automatic differentiation function is based on JAX. '
                    'Please install jax and use `import jax.numpy as jnp` instead '
                    'of `import numpy as np` when defining a combo_fn.')

            # For a general combo_fn F(g_0, g_1, ..., g_k)
            # dF/d θ0,θ1 = sum_i: (∂F/∂g_i)•(d g_i/ d θ0,θ1) + (∂F/∂^2 g_i)•(d g_i/d θ0)•(d g_i/d
            # θ1)

            # term1 = (∂F/∂g_i)•(d g_i/ d θ0,θ1)
        term1 = ListOp([ListOp(operator.oplist, combo_fn=first_partial_combo_fn),
                        ListOp(dd_ops)], combo_fn=lambda x: np.dot(x[1], x[0]))
        # term2 = (∂F/∂^2 g_i)•(d g_i/d θ0)•(d g_i/d θ1)
        term2 = ListOp([ListOp(operator.oplist, combo_fn=second_partial_combo_fn), d1d0_ops],
                       combo_fn=lambda x: np.dot(x[1], x[0]))

        return SummedOp([term1, term2])
