# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
from abc import abstractmethod
from typing import Optional, Union, List

import numpy as np
from scipy.integrate import ode, OdeSolver

from qiskit.algorithms.quantum_time_evolution.evolution_base import EvolutionBase
from qiskit.algorithms.quantum_time_evolution.results.evolution_gradient_result import (
    EvolutionGradientResult,
)
from qiskit.algorithms.quantum_time_evolution.results.evolution_result import EvolutionResult
from qiskit.algorithms.quantum_time_evolution.variational.principles.imaginary.imaginary_variational_principle import (
    ImaginaryVariationalPrinciple,
)
from qiskit.opflow import OperatorBase, Gradient, StateFn
from qiskit.algorithms.quantum_time_evolution.variational.var_qte import VarQte
from qiskit.providers import BaseBackend
from qiskit.utils import QuantumInstance


class VarQite(VarQte, EvolutionBase):
    def __init__(
        self,
        variational_principle: ImaginaryVariationalPrinciple,
        regularization: Optional[str] = None,
        num_time_steps: int = 10,
        init_parameter_values: Optional[Union[List, np.ndarray]] = None,
        ode_solver: Optional[Union[OdeSolver, ode]] = None,
        backend: Optional[Union[BaseBackend, QuantumInstance]] = None,
        snapshot_dir: Optional[str] = None,
        faster: bool = True,
        error_based_ode: bool = False,
        **kwargs,
    ):
        super().__init__(
            variational_principle,
            regularization,
            num_time_steps,
            init_parameter_values,
            ode_solver,
            backend,
            snapshot_dir,
            faster,
            error_based_ode,
            kwargs,
        )

    def evolve(
        self,
        hamiltonian: OperatorBase,
        time: float,
        initial_state: StateFn = None,
        observable: OperatorBase = None,
        t_param=None,
        hamiltonian_value_dict=None,
    ) -> EvolutionResult:
        raise NotImplementedError()

    @abstractmethod
    def gradient(
        self,
        hamiltonian: OperatorBase,
        time: float,
        initial_state: StateFn,
        gradient_object: Gradient,
        observable: OperatorBase = None,
        t_param=None,
        hamiltonian_value_dict=None,
        gradient_params=None,
    ) -> EvolutionGradientResult:
        raise NotImplementedError()
