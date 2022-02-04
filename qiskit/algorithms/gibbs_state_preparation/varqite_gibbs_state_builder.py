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
from typing import Dict, Union, Optional

from qiskit.algorithms.gibbs_state_preparation.default_ansatz_builder import (
    build_ansatz,
    build_init_ansatz_params_vals,
)
from qiskit.algorithms.gibbs_state_preparation.gibbs_state_sampler import GibbsStateSampler
from qiskit.algorithms.gibbs_state_preparation.gibbs_state_builder import GibbsStateBuilder
from qiskit.circuit import Parameter
from qiskit.opflow import OperatorBase
from qiskit.quantum_info import Statevector


class VarQiteGibbsStateBuilder(GibbsStateBuilder):
    """Class for building Gibbs States using Variational Quantum Imaginary Time Evolution
    algorithms."""

    def __init__(
        self,
        qite_algorithm,
    ):
        """
        Args:
            qite_algorithm: Variational Quantum Imaginary Time Evolution algorithm to be used for
                            Gibbs State preparation.
        Raises:
            ValueError: if an ansatz is defined on an odd number of qubits.

        """
        self._qite_algorithm = qite_algorithm
        self._ansatz = None
        self._ansatz_init_params_dict = None

    def _evaluate_initial_ansatz(self) -> Statevector:
        """Binds initial parameters values to an ansatz and returns the result as a state vector."""
        maximally_entangled_states = self._ansatz.assign_parameters(self._ansatz_init_params_dict)
        return Statevector(maximally_entangled_states)

    def build(
        self,
        problem_hamiltonian: OperatorBase,
        temperature: float,
        problem_hamiltonian_param_dict: Optional[Dict[Parameter, Union[complex, float]]] = None,
    ) -> GibbsStateSampler:
        """
        Creates a Gibbs state from given parameters.
        Args:
            problem_hamiltonian: Hamiltonian that defines a desired Gibbs state.
            temperature: Temperature of a desired Gibbs state.
            problem_hamiltonian_param_dict: If a problem Hamiltonian is parametrized, a dictionary
                                            that maps all of its parameters to certain values.
        Returns:
            GibbsState object that includes a relevant quantum state functions as well as
            metadata.
        """
        if self._ansatz is None:  # set default ansatz
            self._set_default_ansatz(problem_hamiltonian)

        time = 1 / (2 * self.BOLTZMANN_CONSTANT * temperature)

        param_dict = {**self._ansatz_init_params_dict, **problem_hamiltonian_param_dict}
        gibbs_state_function = self._qite_algorithm.evolve(
            hamiltonian=problem_hamiltonian,
            time=time,
            initial_state=self._ansatz,
            hamiltonian_value_dict=param_dict,
        )

        aux_registers = set(range(self._ansatz.num_qubits / 2, self._ansatz.num_qubits))

        return GibbsStateSampler(
            gibbs_state_function=gibbs_state_function,
            hamiltonian=problem_hamiltonian,
            temperature=temperature,
            ansatz=self._ansatz,
            ansatz_params_dict=None,  # TODO get from evolution result
            ansatz_hamiltonian_gradients=None,  # TODO get from evolution result
            aux_registers=aux_registers,
        )

    def _set_default_ansatz(self, problem_hamiltonian: OperatorBase) -> None:
        """
        Sets a default ansatz with default parameters for a Gibbs state preparation.
        Args:
            problem_hamiltonian: Hamiltonian that defines a desired Gibbs state.
        """
        num_qubits = problem_hamiltonian.num_qubits
        depth = 1
        self._ansatz = build_ansatz(num_qubits, depth)
        ansatz_init_params_vals = build_init_ansatz_params_vals(num_qubits, depth)
        self._ansatz_init_params_dict = dict(
            zip(self._ansatz.ordered_parameters, ansatz_init_params_vals)
        )
