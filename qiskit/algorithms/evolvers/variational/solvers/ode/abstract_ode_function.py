# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Abstract class for generating ODE functions."""

from abc import ABC, abstractmethod
from typing import Iterable, Dict, Optional
from qiskit.circuit import Parameter
from ..var_qte_linear_solver import (
    VarQTELinearSolver,
)


class AbstractOdeFunction(ABC):
    """Abstract class for generating ODE functions."""

    def __init__(
        self,
        varqte_linear_solver: VarQTELinearSolver,
        error_calculator,
        param_dict: Dict[Parameter, complex],
        t_param: Optional[Parameter] = None,
    ) -> None:

        self._varqte_linear_solver = varqte_linear_solver
        self._error_calculator = error_calculator
        self._param_dict = param_dict
        self._t_param = t_param

    @abstractmethod
    def var_qte_ode_function(self, time: float, parameters_values: Iterable) -> Iterable:
        """
        Evaluates an ODE function for a given time and parameter values. It is used by an ODE
        solver.

        Args:
            time: Current time of evolution.
            parameters_values: Current values of parameters.

        Returns:
            ODE gradient arising from solving a system of linear equations.
        """
        pass
