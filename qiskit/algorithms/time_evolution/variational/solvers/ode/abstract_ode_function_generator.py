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
from typing import Iterable, Union, Dict, Optional

from qiskit.algorithms.time_evolution.variational.error_calculators.gradient_errors \
    .error_calculator import \
    ErrorCalculator
from qiskit.algorithms.time_evolution.variational.variational_principles.variational_principle \
    import (
    VariationalPrinciple,
)
from qiskit.algorithms.time_evolution.variational.solvers.var_qte_linear_solver import (
    VarQteLinearSolver,
)
from qiskit.circuit import Parameter


class AbstractOdeFunctionGenerator(ABC):
    """Abstract class for generating ODE functions."""

    def __init__(
        self,
        regularization: Optional[str] = None,
    ):
        """
        Args:
            regularization: Use the following regularization with a least square method to solve the
                underlying system of linear equations. Can be either None or ``'ridge'`` or
                ``'lasso'`` or ``'perturb_diag'``. ``'ridge'`` and ``'lasso'`` use an automatic
                optimal parameter search. If regularization is None but the metric is
                ill-conditioned or singular then a least square solver is used without
                regularization.
        """
        self._regularization = regularization

    def _lazy_init(
        self,
        error_calculator: ErrorCalculator,
        variational_principle: VariationalPrinciple,
        t_param: Parameter,
        param_dict: Dict[Parameter, Union[float, complex]],
        linear_solver: VarQteLinearSolver,
    ):
        self._error_calculator = error_calculator
        self._variational_principle = variational_principle
        self._t_param = t_param
        self._param_dict = param_dict
        self._linear_solver = linear_solver

    @abstractmethod
    def var_qte_ode_function(self, time: float, parameters_values: Iterable) -> Iterable:
        """
        Evaluates an ODE function for a given time and parameter values. It is used by an ODE
        solver.

        Args:
            time: Current time of evolution.
            parameters_values: Current values of parameters.
        Returns:

            Tuple containing natural gradient, metric tensor and evolution gradient results
            arising from solving a system of linear equations.
        """
        pass
