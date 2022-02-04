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
"""Class representing a quantum state of a Gibbs State along with metadata and gradient
calculation methods."""
from typing import Optional, Union, Dict, Set

import numpy as np
import numpy.typing
from numpy.typing import NDArray

from qiskit.circuit import Parameter
from qiskit.opflow import StateFn, OperatorBase, Gradient, CircuitStateFn, CircuitSampler
from qiskit.providers import Backend, BaseBackend
from qiskit.utils import QuantumInstance


class GibbsStateSampler:
    """Class representing a quantum state of a Gibbs State along with metadata and gradient
    calculation methods."""

    def __init__(
        self,
        gibbs_state_function: StateFn,
        hamiltonian: Optional[OperatorBase],
        temperature: Optional[float],
        ansatz: Optional[OperatorBase] = None,
        ansatz_params_dict: Optional[Dict[Parameter, Union[complex, float]]] = None,
        ansatz_hamiltonian_gradients: Optional[
            Dict[Parameter, np.typing.NDArray[Union[complex, float]]]
        ] = None,
        aux_registers: Set[int] = None,
    ):
        """
        Args:
            gibbs_state_function: Quantum state function of a Gibbs state.
            hamiltonian: Hamiltonian used to build a Gibbs state.
            temperature: Temperature used to build a Gibbs state.
            ansatz: Ansatz that gave rise to a Gibbs state.
            ansatz_params_dict: Dictionary that maps ansatz parameters to values optimal for a
                                Gibbs state.
            ansatz_hamiltonian_gradients: Gradients of ansatz parameters w.r.t. Hamiltonian
                                          parameters. The dictionary key is assumed to be a
                                          Hamiltonian parameter. It gives access to a numpy array
                                          of ansatz parameters gradients w.r.t. that Hamiltonian
                                          parameter. The numpy array is assumed to be ordered in the
                                          same way as parameters are ordered in an ansatz (i.e.
                                          according to ansatz.ordered_parameters). Needed for
                                          Variational Quantum Boltzmann Machine algorithm for
                                          example. Might be obtained from a Variational Quantum
                                          Imaginary Time Evolution algorithm for example.
            aux_registers: Set of indices (0-indexed) of registers in an ansatz that are auxiliary,
                            i.e. they do not contain a Gibbs state. E.g. in VarQiteGibbsStateBuilder
                            the second half or registers is auxiliary.
        """
        self._gibbs_state_function = gibbs_state_function
        self._hamiltonian = hamiltonian
        self._temperature = temperature
        self._ansatz = ansatz
        self._ansatz_params_dict = ansatz_params_dict
        self._ansatz_hamiltonian_gradients = ansatz_hamiltonian_gradients
        self._aux_registers = aux_registers

    @property
    def hamiltonian(self) -> OperatorBase:
        """Returns a hamiltonian."""
        return self._hamiltonian

    @property
    def temperature(self) -> float:
        """Returns a temperature."""
        return self._temperature

    @property
    def ansatz(self) -> OperatorBase:
        """Returns an ansatz that gave rise to a Gibbs state."""
        return self._ansatz

    @property
    def ansatz_params_dict(self) -> Dict[Parameter, Union[complex, float]]:
        """Returns a dictionary that maps ansatz parameters to values that gave rise to a Gibbs
        state."""
        return self._ansatz_params_dict

    @property
    def aux_registers(self) -> Set[int]:
        """Returns a set of indices of qubits that are auxiliary registers in an ansatz and do
        not contain the Gibbs state."""
        return self._aux_registers

    def eval_gibbs_state_matrix(self):
        """Evaluates a Gibbs state matrix on a given backend. Note that this process is generally
        not efficient and should not be used in production settings."""
        pass

    # TODO caching through sampler?
    def sample(
        self, backend: Union[Backend, BaseBackend, QuantumInstance]
    ) -> NDArray[Union[complex, float]]:  # calc p_qbm
        """
        Samples probabilities from a Gibbs state.
        Args:
            backend: A backend used for sampling circuit probabilities.
        Returns:
            An array of samples probabilities.
        """
        operator = CircuitStateFn(self._ansatz)
        sampler = CircuitSampler(backend=backend).convert(operator, self._ansatz_params_dict)
        amplitudes_with_aux_regs = sampler.eval().primitive
        probs = self._discard_aux_registers(amplitudes_with_aux_regs)
        return probs

    def calc_ansatz_gradients(
        self,
        backend: Union[Backend, BaseBackend, QuantumInstance],
        gradient_method: str = "param_shift",
    ) -> NDArray[NDArray[Union[complex, float]]]:
        """
        Calculates gradients of a Gibbs state w.r.t. desired
        gradient_params that parametrize the Gibbs state.
        Args:
            backend: A backend used for sampling circuit probabilities.
            gradient_method: A desired gradient method chosen from the Qiskit Gradient Framework.
        Returns:
            Calculated gradients with respect to each parameter indicated in gradient_params
            with bound parameter values.
        Raises:
            ValueError: If ansatz and ansatz_params_dict are not both provided.
        """
        if not self._ansatz or not self._ansatz_params_dict:
            raise ValueError(
                "Both ansatz and ansatz_params_dict must be present in the class to compute "
                "gradients."
            )
        gradient_params = list(self._ansatz_params_dict.keys())

        operator = CircuitStateFn(self._ansatz)
        # combo fn for an operator here, telling how to batch results together, pass combo fn to
        # Gradient
        # state_grad_with_aux_regs = Gradient(grad_method=gradient_method).convert(
        #     operator=operator, params=gradient_params
        # )
        # sampler = CircuitSampler(backend=backend).convert(
        #     state_grad_with_aux_regs, self._ansatz_params_dict
        # )
        # gradient_amplitudes_with_aux_regs = sampler.eval()[0]
        gradient_amplitudes_with_aux_regs = Gradient(grad_method=gradient_method).gradient_wrapper(
            operator, self._ansatz.ordered_parameters, backend=backend
        )
        # Get the values for the gradient of the sampling probabilities w.r.t. the Ansatz parameters
        gradient_amplitudes_with_aux_regs = gradient_amplitudes_with_aux_regs(
            self.ansatz_params_dict.values()
        )
        # TODO gradients of amplitudes or probabilities?
        state_grad = self._discard_aux_registers_gradients(gradient_amplitudes_with_aux_regs)
        return state_grad

    def calc_hamiltonian_gradients(
        self,
        backend: Union[Backend, BaseBackend, QuantumInstance],
        gradient_method: str = "param_shift",
    ) -> Dict[Parameter, NDArray[Union[complex, float]]]:
        """
        Calculates gradients of the visible part of a Gibbs state w.r.t. parameters of a
        Hamiltonian that gave rise to the Gibbs state.
        Args:
            backend: A backend used for sampling circuit probabilities.
            gradient_method: A desired gradient method chosen from the Qiskit Gradient Framework.
        Returns:
            Calculated gradients of the visible part of a Gibbs state w.r.t. parameters of a
            Hamiltonian that gave rise to the Gibbs state.
        """
        ansatz_gradients = self.calc_ansatz_gradients(backend, gradient_method)
        gibbs_state_hamiltonian_gradients = {}
        for hamiltonian_parameter in self._hamiltonian.parameters:
            summed_gradient = np.dot(
                ansatz_gradients, self._ansatz_hamiltonian_gradients[hamiltonian_parameter]
            )

            gibbs_state_hamiltonian_gradients[hamiltonian_parameter] = summed_gradient

        return gibbs_state_hamiltonian_gradients

    def _discard_aux_registers(
        self, amplitudes_with_aux_regs: NDArray[Union[complex, float]]
    ) -> NDArray[Union[complex, float]]:
        """
        Accepts an object with complex amplitudes sampled from a state with auxiliary
        registers and processes bit strings of qubit labels. For the default choice of an ansatz
        in the GibbsStateBuilder, this method gets rid of the second half of qubits that
        correspond to an auxiliary system. Then, it aggregates complex amplitudes and returns the
        vector of probabilities. Indices of returned probability vector correspond to labels of a
        reduced qubit state.
        Args:
            amplitudes_with_aux_regs: An array of amplitudes sampled from a Gibbs state circuit
                                        that includes auxiliary registers and their measurement
                                        outcomes.
        Returns:
            An array of probability samples from a Gibbs state only (excluding auxiliary registers).
        Raises:
            ValueError: If a provided number of qubits for an ansatz is not even.
        """
        kept_num_qubits = self._ansatz.num_qubits - len(self._aux_registers)

        amplitudes = amplitudes_with_aux_regs.data
        amplitudes_qubit_labels_ints = amplitudes_with_aux_regs.indices
        reduced_qubits_amplitudes = np.zeros(pow(2, kept_num_qubits))

        for qubit_label_int, amplitude in zip(amplitudes_qubit_labels_ints, amplitudes):
            reduced_label = self._reduce_label(qubit_label_int)
            reduced_qubits_amplitudes[reduced_label] += np.conj(amplitude) * amplitude

        return reduced_qubits_amplitudes

    def _reduce_label(self, label: int) -> int:
        cnt = len(bin(label)) - 2
        cnt2 = 0
        reduced_label_bits = []
        while cnt:
            bit = label & 1
            label = label >> 1
            if cnt2 not in self._aux_registers:
                reduced_label_bits.append(bit)
            cnt -= 1
            cnt2 += 1
        reduced_label = 0
        for bit in reduced_label_bits[::-1]:
            reduced_label = (reduced_label << 1) | bit
        return reduced_label

    def _discard_aux_registers_gradients(
        self, amplitudes_with_aux_regs: NDArray[Union[complex, float]]
    ) -> NDArray[NDArray[Union[complex, float]]]:
        """
        Accepts an object with complex amplitude gradients sampled from a state with auxiliary
        registers and processes bit strings of qubit labels. For the default choice of an
        ansatz in the GibbsStateBuilder, this method gets rid of the second half of qubits that
        correspond to an auxiliary system. Then, it aggregates complex amplitudes gradients and
        returns the vector of probability gradients. Indices of returned probability gradients
        vector correspond to labels of a reduced qubit state.
        Args:
            amplitudes_with_aux_regs: An array of amplitudes gradients sampled and calculated from
                                        a Gibbs state circuit that includes auxiliary registers and
                                        their measurement outcomes.
        Returns:
            An array of probability gradients from a Gibbs state only (excluding auxiliary
            registers).
        """
        reduced_qubits_amplitudes = np.zeros(len(amplitudes_with_aux_regs), dtype=object)
        for ind, amplitude_data in enumerate(amplitudes_with_aux_regs):
            res = self._discard_aux_registers(amplitude_data)
            reduced_qubits_amplitudes[ind] = res

        return reduced_qubits_amplitudes
