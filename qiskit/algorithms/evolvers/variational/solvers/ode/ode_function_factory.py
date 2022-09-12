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
from typing import Dict, Any, Optional, Callable

import numpy as np

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
    """Factory for building ODE functions."""

    def __init__(
        self,
        ode_function_type: OdeFunctionType = OdeFunctionType.STANDARD_ODE,
        lse_solver: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    ) -> None:
        """
        Args:
            ode_function_type: An Enum that defines a type of an ODE function to be built. If
                not provided, a default ``STANDARD_ODE`` is used.
            lse_solver: Linear system of equations solver callable. It accepts ``A`` and ``b`` to
                solve ``Ax=b`` and returns ``x``. If ``None``, the default ``np.linalg.lstsq``
                solver is used.
        """
        self.ode_function_type = ode_function_type
        self.lse_solver = lse_solver

    def _build(
        self,
        varqte_linear_solver: VarQTELinearSolver,
        error_calculator: Any,
        param_dict: Dict[Parameter, complex],
        t_param: Optional[Parameter] = None,
    ) -> AbstractOdeFunction:
        """
        Initializes an ODE function specified in the class.

        Args:
            varqte_linear_solver: Solver of LSE for the VarQTE algorithm.
            error_calculator: Calculator of errors for error-based ODE functions.
            param_dict: Dictionary which relates parameter values to the parameters in the ansatz.
            t_param: Time parameter in case of a time-dependent Hamiltonian.

        Returns:
            An ODE function.

        Raises:
            ValueError: If unsupported ODE function provided.

        """
        if self.ode_function_type == OdeFunctionType.STANDARD_ODE:
            return OdeFunction(varqte_linear_solver, error_calculator, param_dict, t_param)
        raise ValueError(
            f"Unsupported ODE function provided: {self.ode_function_type}."
            f" Only {[tp.value for tp in OdeFunctionType]} are supported."
        )
