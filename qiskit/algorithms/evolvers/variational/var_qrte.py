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

"""Variational Quantum Real Time Evolution algorithm."""
from functools import partial
from typing import Optional, Callable, Union

import numpy as np
from scipy.integrate import OdeSolver

from qiskit.algorithms.evolvers.real_evolver import RealEvolver
from qiskit.opflow import (
    ExpectationBase,
)
from qiskit.utils import QuantumInstance
from .solvers.ode.ode_function_factory import OdeFunctionFactory
from .var_qte import VarQTE
from .variational_principles.real_variational_principle import (
    RealVariationalPrinciple,
)


class VarQRTE(VarQTE, RealEvolver):
    """Variational Quantum Real Time Evolution algorithm."""

    def __init__(
        self,
        variational_principle: RealVariationalPrinciple,
        ode_function_factory: OdeFunctionFactory,
        ode_solver: Union[OdeSolver, str] = "RK45",
        lse_solver: Callable[[np.ndarray, np.ndarray], np.ndarray] = partial(
            np.linalg.lstsq, rcond=1e-2
        ),
        expectation: Optional[ExpectationBase] = None,
        imag_part_tol: float = 1e-7,
        num_instability_tol: float = 1e-7,
        quantum_instance: Optional[QuantumInstance] = None,
    ) -> None:
        r"""
        Args:
            variational_principle: Variational Principle to be used.
            ode_function_factory: Factory for the ODE function.
            ode_solver: ODE solver callable that implements a SciPy ``OdeSolver`` interface or a
                string indicating a valid method offered by SciPy.
            lse_solver: Linear system of equations solver that follows a NumPy
                ``np.linalg.lstsq`` interface.
            expectation: An instance of ``ExpectationBase`` which defines a method for calculating
                expectation values of ``EvolutionProblem.aux_operators``.
            imag_part_tol: Allowed value of an imaginary part that can be neglected if no
                imaginary part is expected.
            num_instability_tol: The amount of negative value that is allowed to be
                rounded up to 0 for quantities that are expected to be
                non-negative.
            quantum_instance: Backend used to evaluate the quantum circuit outputs. If ``None``
                provided, everything will be evaluated based on matrix multiplication (which is
                slow).
        """
        super().__init__(
            variational_principle,
            ode_function_factory,
            ode_solver,
            lse_solver,
            expectation,
            imag_part_tol,
            num_instability_tol,
            quantum_instance,
        )
