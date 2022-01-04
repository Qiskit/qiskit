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
"""Class for building Gibbs States using Quantum Imaginary Time Evolution algorithms."""
from typing import Dict, Union

from qiskit import QuantumCircuit
from qiskit.algorithms.gibbs_state_preparation.gibbs_state import GibbsState
from qiskit.algorithms.gibbs_state_preparation.gibbs_state_builder import GibbsStateBuilder
from qiskit.circuit import Parameter
from qiskit.opflow import OperatorBase
from qiskit.providers import BaseBackend
from qiskit.quantum_info import state_fidelity, Statevector
from qiskit.utils import QuantumInstance


class QiteGibbsStateBuilder(GibbsStateBuilder):
    """Class for building Gibbs States using Quantum Imaginary Time Evolution algorithms."""

    def __init__(
        self,
        qite_algorithm,
        ansatz: OperatorBase,
        ansatz_init_params_dict: Dict[Parameter, Union[complex, float]],
    ):
        """
        Args:
            qite_algorithm: Quantum Imaginary Time Evolution algorithm to be used for Gibbs State
                            preparation.
            ansatz: Initial ansatz for the qite_algorithm. Together with ansatz_init_params_dict,
                    it should result in n Maximally Entangled States, where n is half the number of
                    qubits in an ansatz.
            ansatz_init_params_dict: Dictionary that maps parameters from ansatz to their initial
                                     values. When bound to an ansatz, it should result in n
                                     Maximally Entangled States, where n is half the number of
                                     qubits in an ansatz.
        Raises:
            ValueError: if an ansatz is defined on an odd number of qubits.

        """
        self._qite_algorithm = qite_algorithm
        if ansatz.num_qubits % 2 != 0:
            raise ValueError(
                f"QiteGibbsStateBuilder requires an ansatz on an even number of qubits. "
                f"{ansatz.num_qubits} qubits provided."
            )  # TODO might be specific to VarQite?
        self._ansatz = ansatz
        self._ansatz_init_params_dict = ansatz_init_params_dict
        # TODO ansatz and init params need to form n MES, add a check? add defaults?

    def _evaluate_initial_ansatz(self) -> Statevector:
        """Binds initial parameters values to an ansatz and returns the result as a state vector."""
        maximally_entangled_states = self._ansatz.assign_parameters(self._ansatz_init_params_dict)
        return Statevector(maximally_entangled_states)

    def _build_n_mes(self, num_qubits, backend: Union[BaseBackend, QuantumInstance]) -> Statevector:
        """Builds n Maximally Entangled States (MES) as state vectors exactly."""
        qc = QuantumCircuit(2)
        for _ in range(num_qubits):
            qc = qc @ self._build_mes()

        return backend.run(qc).result().get_statevector()

    def _build_mes(self) -> QuantumCircuit:
        """Builds a quantum circuit for a single Maximally Entangled State (MES)."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        return qc

    # TODO or by tracing out to the maximally mixed state?
    def _calc_ansatz_mes_fidelity(self, backend: Union[BaseBackend, QuantumInstance]) -> float:
        """Calculates fidelity between n exact Maximally Entangled States (MES) and bound ansatz."""
        num_of_mes = self._ansatz.num_qubits / 2
        exact_n_mes = self._build_n_mes(num_of_mes, backend)
        ansatz_n_mes = self._evaluate_initial_ansatz()
        return state_fidelity(exact_n_mes, ansatz_n_mes)

    def build(self, problem_hamiltonian: OperatorBase, temperature: float) -> GibbsState:
        """
        Creates a Gibbs state from given parameters.
        Args:
            problem_hamiltonian: Hamiltonian that defines a desired Gibbs state.
            temperature: Temperature of a desired Gibbs state.
        Returns:
            GibbsState object that includes a relevant quantum state functions as well as
            metadata.
        """
        time = 1 / (self.BOLTZMANN_CONSTANT * temperature)
        gibbs_state_function = self._qite_algorithm.evolve(
            hamiltonian=problem_hamiltonian,
            time=time,
            initial_state=self._ansatz,
            hamiltonian_value_dict=self._ansatz_init_params_dict,
        )
        return GibbsState(
            gibbs_state_function=gibbs_state_function,
            hamiltonian=problem_hamiltonian,
            temperature=temperature,
        )
