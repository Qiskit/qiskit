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

from typing import Union

from qiskit.aqua.operators.gradients.circuit_gradients.circuit_gradient \
    import CircuitGradient
from qiskit.aqua.operators.gradients.derivative_base import DerivativeBase


class HessianBase(DerivativeBase):  # pylint: disable=abstract-method
    """Base class for the Hessian of an expected value."""

    def __init__(self,
                 hess_method: Union[str, CircuitGradient] = 'param_shift',
                 **kwargs):
        r"""
        Args:
            hess_method: The method used to compute the state/probability gradient. Can be either
                         ``'param_shift'`` or ``'lin_comb'`` or ``'fin_diff'``.
                         Ignored for gradients w.r.t observable parameters.
            kwargs (dict): Optional parameters for a CircuitGradient

        Raises:
            ValueError: If method != ``fin_diff`` and ``epsilon`` is not None.
        """

        if isinstance(hess_method, CircuitGradient):
            self._hess_method = hess_method
        elif hess_method == 'param_shift':
            from .circuit_gradients import ParamShift
            self._hess_method = ParamShift()

        elif hess_method == 'fin_diff':
            from .circuit_gradients import ParamShift
            epsilon = kwargs.get('epsilon', 1e-6)
            self._hess_method = ParamShift(analytic=False, epsilon=epsilon)

        elif hess_method == 'lin_comb':
            from .circuit_gradients import LinComb
            self._hess_method = LinComb()

        else:
            raise ValueError("Unrecognized input provided for `hess_method`. Please provide"
                             " a CircuitGradient object or one of the pre-defined string"
                             " arguments: {'param_shift', 'fin_diff', 'lin_comb'}. ")

    @property
    def hess_method(self) -> CircuitGradient:
        """Returns ``CircuitGradient``.

        Returns:
            ``CircuitGradient``.

        """
        return self._hess_method
