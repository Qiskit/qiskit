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
from typing import Iterable, Union, Dict, Callable

import numpy as np

from qiskit.algorithms.time_evolution.variational.error_calculators.gradient_errors.error_calculator import (
    ErrorCalculator,
)
from qiskit.algorithms.time_evolution.variational.solvers.var_qte_linear_solver import (
    VarQteLinearSolver,
)
from qiskit.circuit import Parameter


class AbstractOdeFunctionGenerator(ABC):
    """Abstract class for generating ODE functions."""

    def __init__(
        self,
        optimizer: str = "COBYLA",
        optimizer_tolerance: float = 1e-6,
    ):
        """
        Args:
            optimizer:
            optimizer_tolerance:
        """

        self._optimizer = optimizer
        self._optimizer_tolerance = optimizer_tolerance

        self._varqte_linear_solver = None
        self._t_param = None
        self._param_dict = None

    def _lazy_init(
        self,
        varqte_linear_solver: VarQteLinearSolver,
        error_calculator: ErrorCalculator,
        t_param: Parameter,
        param_dict: Dict[Parameter, Union[float, complex]],
    ):
        self._varqte_linear_solver = varqte_linear_solver
        self._error_calculator = error_calculator
        self._t_param = t_param
        self._param_dict = param_dict

    @abstractmethod
    def var_qte_ode_function(self, time: float, parameters_values: Iterable) -> Iterable:
        """
        Evaluates an ODE function for a given time and parameter values. It is used by an ODE
        solver.

        Args:
            time: Current time of evolution.
            parameters_values: Current values of parameters.

        Returns:
            Natural gradient arising from solving a system of linear equations.
        """
        pass
