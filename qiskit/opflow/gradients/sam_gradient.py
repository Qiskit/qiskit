
"""The base interface for Opflow's gradient."""

from functools import partial
from typing import Union, List, Optional

import numpy as np

from qiskit.circuit import ParameterExpression, ParameterVector
from .gradient import Gradient
from .hessian import Hessian
from ..list_ops.list_op import ListOp
from ..operator_base import OperatorBase
from ..operator_globals import Zero

try:
    from jax import grad, jit
    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False


class SAMGradient(Gradient):

    def __init__(self, rho=0.05, second_order=False):
        self._rho = rho
        self._second_order = second_order
        super().__init__()

    def convert(self,
                operator: OperatorBase,
                params: Optional[
                    Union[ParameterVector, ParameterExpression, List[ParameterExpression]]] = None
                ) -> OperatorBase:

        params_op = ListOp([param * (~Zero @ Zero) for param in params])

        grad_op = Gradient().convert(operator, params)

        if self._second_order:
            hess_op = Hessian().convert(operator, params)
            return ListOp([grad_op, hess_op, params_op],
                          combo_fn=partial(self._second_ord, grad_op=grad_op, params=params))
        else:
            return ListOp([grad_op, params_op],
                          combo_fn=partial(self._first_ord, grad_op=grad_op, params=params))

    def _first_ord(self, values, grad_op, params):
        grad_values = np.real(values[0])
        param_values = np.real(values[1])

        eps = self._rho * grad_values / np.linalg.norm(grad_values)

        param_values += eps  # shift param values by eps

        param_dict = {param: param_values[i] for i, param in enumerate(params)}

        return grad_op.bind_parameters(param_dict).eval()  # 1st order approx of SAM gradient

    def _second_ord(self, values, grad_op, params):
        grad_values = np.real(values[0])
        hess_values = np.real(values[1])
        param_values = np.real(values[2])

        sam_grad = self._first_ord([grad_values, param_values], grad_op, params)  # 1st order approx

        norm_grad = np.linalg.norm(grad_values)  # precompute gradient norm

        # gradient of eps
        d_eps = self._rho * hess_values / norm_grad \
            - self._rho * np.outer(hess_values.dot(grad_values), grad_values) / (norm_grad ** 3)

        return sam_grad + d_eps.dot(sam_grad)  # 2nd order approx of SAM gradient
