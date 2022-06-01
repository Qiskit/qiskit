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

from abc import ABC
from enum import Enum
from typing import Dict, Any
from qiskit.circuit import Parameter
from .abstract_ode_function import AbstractOdeFunction
from .ode_function import OdeFunction
from ..var_qte_linear_solver import (
    VarQTELinearSolver,
)


class OdeFunctionType(Enum):
    """Types of ODE functions for VatQTE algorithms."""

    # more will be supported in the near future
    STANDARD_ODE = "STANDARD_ODE"


class OdeFunctionFactory(ABC):
    """Abstract class for generating ODE functions."""

    def __init__(self, ode_function_type: OdeFunctionType) -> None:

        self._ode_type = ode_function_type

    def build(
        self,
        varqte_linear_solver: VarQTELinearSolver,
        error_calculator: Any,  # TODO will be supported in another PR
        t_param: Parameter,
        param_dict: Dict[Parameter, complex],
    ) -> AbstractOdeFunction:
        """
        Initializes an ODE function specified in the class.

        Args:
            varqte_linear_solver: Solver of LSE for the VarQTE algorithm.
            error_calculator: Calculator of errors for error-based ODE functions.
            t_param: Time parameter in case of a time-dependent Hamiltonian.
            param_dict: Dictionary which relates parameter values to the parameters in the ansatz.
        Returns:
            An ODE function.
        Raises:
            ValueError: If unsupported ODE function provided.

        """
        if self._ode_type == OdeFunctionType.STANDARD_ODE:
            return OdeFunction(varqte_linear_solver, error_calculator, t_param, param_dict)
        raise ValueError(
            f"Unsupported ODE function provided: {self._ode_type}. Only {[tp.value for tp in OdeFunctionType]} are supported."
        )
