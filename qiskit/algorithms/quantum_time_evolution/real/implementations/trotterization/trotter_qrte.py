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
from qiskit.algorithms.quantum_time_evolution.builders.implementations.trotterizations.suzuki import (
    Suzuki,
)
from qiskit.algorithms.quantum_time_evolution.builders.implementations.trotterizations.trotter_mode_enum import (
    TrotterModeEnum,
)
from qiskit.algorithms.quantum_time_evolution.builders.implementations.trotterizations.trotterization_factory import (
    TrotterizationFactory,
)
from qiskit.algorithms.quantum_time_evolution.real.qrte import Qrte
from qiskit.algorithms.quantum_time_evolution.results.evolution_gradient_result import (
    EvolutionGradientResult,
)
from qiskit.algorithms.quantum_time_evolution.results.evolution_result import EvolutionResult
from qiskit.circuit import Parameter
from qiskit.opflow import OperatorBase, StateFn, Gradient


class TrotterQrte(Qrte):
    def __init__(self, mode: TrotterModeEnum, reps: int = 1):
        self._mode = mode
        self._reps = reps

    def evolve(
        self,
        hamiltonian: OperatorBase,
        time: float,
        initial_state: StateFn = None,
        observable: OperatorBase = None,
        t_param: Parameter = None,
        hamiltonian_value_dict=None,
    ) -> EvolutionResult:

        hamiltonian = self._try_binding_params(hamiltonian, hamiltonian_value_dict, t_param)

        trotter = TrotterizationFactory.build(self._mode, self._reps)
        trotterized_hamiltonian = trotter.build(time * hamiltonian)
        self._validate_input(initial_state, observable)

        if initial_state is not None:
            return (trotterized_hamiltonian @ initial_state).eval()
        if observable is not None:
            return trotterized_hamiltonian.adjoint() @ observable @ trotterized_hamiltonian

    def _try_binding_params(self, hamiltonian, hamiltonian_value_dict, t_param):
        if t_param is not None:
            if hamiltonian_value_dict is None:
                raise ValueError(
                    "t_param provided indicating a parametrized Hamiltonian but no "
                    "hamiltonian_value_dict provided. Parameters need to be bindable to perform "
                    "evolution."
                )
            else:
                hamiltonian = hamiltonian.bind_parameters(hamiltonian_value_dict)
        return hamiltonian

    def _validate_input(self, initial_state, observable):
        if initial_state is None and observable is None:
            raise ValueError(
                "TrotterQrte requires an initial state or an observable to be evolved; None "
                "provided."
            )
        elif initial_state is not None and observable is not None:
            raise ValueError(
                "TrotterQrte requires an initial state or an observable to be evolved; both "
                "provided."
            )

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


