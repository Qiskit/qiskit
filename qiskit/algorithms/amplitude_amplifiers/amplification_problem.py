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

"""The Amplification problem class."""

from typing import Optional, Callable, Any, Union, List

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import GroverOperator
from qiskit.quantum_info import Statevector


class AmplificationProblem:
    """The amplification problem is the input to amplitude amplification algorithms, like Grover.

    This class contains all problem-specific information required to run an amplitude amplification
    algorithm. It minimally contains the Grover operator. It can further hold some post processing
    on the optimal bitstring.
    """

    def __init__(self,
                 oracle: Union[QuantumCircuit, Statevector],
                 state_preparation: Optional[QuantumCircuit] = None,
                 grover_operator: Optional[QuantumCircuit] = None,
                 post_processing: Optional[Callable[[str], Any]] = None,
                 objective_qubits: Optional[Union[int, List[int]]] = None,
                 is_good_state: Optional[Union[
                     Callable[[str], bool], List[int], List[str], Statevector]] = None,
                 ) -> None:
        r"""
        Args:
            oracle: The oracle reflecting about the bad states.
            state_preparation: A circuit preparing the input state, referred to as
                :math:`\mathcal{A}`. If None, a layer of Hadamard gates is used.
            grover_operator: The Grover operator :math:`\mathcal{Q}` used as unitary in the
                phase estimation circuit. If None, this operator is constructed from the ``oracle``
                and ``state_preparation``.
            post_processing: A mapping applied to the most likely bitstring.
            objective_qubits: If set, specifies the indices of the qubits that should be measured.
                If None, all qubits will be measured. The ``is_good_state`` function will be
                applied on the measurement outcome of these qubits.
            is_good_state: A function to check whether a string represents a good state.
        """
        self.oracle = oracle

        if state_preparation:
            self.state_preparation = state_preparation
        else:
            self.state_preparation = QuantumCircuit(oracle.num_qubits)
            self.state_preparation.h(range(oracle.num_qubits))

        self.grover_operator = grover_operator or GroverOperator(oracle, self.state_preparation)
        self.post_processing = post_processing or (lambda x: x)
        self._objective_qubits = objective_qubits
        self._is_good_state = is_good_state

    @property
    def objective_qubits(self) -> List[int]:
        """The indices of the objective qubits.

        Returns:
            The indices of the objective qubits as list of integers.
        """
        if self._objective_qubits is None:
            return list(range(self.oracle.num_qubits))

        if isinstance(self._objective_qubits, int):
            return [self._objective_qubits]

        return self._objective_qubits

    @objective_qubits.setter
    def objective_qubits(self, objective_qubits: Optional[Union[int, List[int]]]) -> None:
        """Set the objective qubits.

        Args:
            objective_qubits: The indices of the qubits that should be measured.
                If None, all qubits will be measured. The ``is_good_state`` function will be
                applied on the measurement outcome of these qubits.
        """
        self._objective_qubits = objective_qubits

    @property
    def is_good_state(self) -> Callable[[str], bool]:
        """Check whether a provided bitstring is a good state or not.

        Returns:
            A callable that takes in a bitstring and returns True if the measurement is a good
            state, False otherwise.
        """
        if callable(self._is_good_state):
            return self._is_good_state
        elif isinstance(self._is_good_state, list):
            if all(isinstance(good_bitstr, str) for good_bitstr in self._is_good_state):
                return lambda bitstr: bitstr in self._is_good_state
            else:
                return lambda bitstr: all(bitstr[good_index] == '1'  # type:ignore
                                          for good_index in self._is_good_state)

        return lambda bitstr: bitstr in self._is_good_state.probabilities_dict()

    @is_good_state.setter
    def is_good_state(self,
                      is_good_state: Union[Callable[[str], bool], List[int], List[str], Statevector]
                      ) -> None:
        """Set the ``is_good_state`` function.

        Args:
            is_good_state: A function to determine whether a bitstring represents a good state.
        """
        self._is_good_state = is_good_state
