# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Estimation problem class."""

from typing import Optional, List, Callable

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import GroverOperator


class EstimationProblem:
    """The estimation problem is the input to amplitude estimation algorithm.

    This class contains all problem-specific information required to run an amplitude estimation
    algorithm. That means, it minimally contains the state preparation and the specification
    of the good state. It can further hold some post processing on the estimation of the amplitude
    or a custom Grover operator.
    """

    def __init__(self,
                 state_preparation: Optional[QuantumCircuit] = None,
                 grover_operator: Optional[QuantumCircuit] = None,
                 objective_qubits: Optional[List[int]] = None,
                 post_processing: Optional[Callable[[float], float]] = None,
                 is_good_state: Optional[Callable[[str], bool]] = None,
                 ) -> None:
        r"""
        Args:
            state_preparation: A circuit preparing the input state, referred to as
                :math:`\mathcal{A}`.
            grover_operator: The Grover operator :math:`\mathcal{Q}` used as unitary in the
                phase estimation circuit.
            objective_qubits: A list of qubit indices to specify the oracle in the Grover operator,
                if the Grover operator is not supplied. A measurement outcome is classified as
                'good' state if all objective qubits are in state :math:`|1\rangle`, otherwise it
                is classified as 'bad'.
            post_processing: A mapping applied to the result of the algorithm
                :math:`0 \leq a \leq 1`, usually used to map the estimate to a target interval.
            is_good_state: A function to check whether a string represents a good state.
        """
        self._state_preparation = state_preparation
        self._grover_operator = grover_operator
        self._objective_qubits = objective_qubits
        self._post_processing = post_processing
        self._is_good_state = is_good_state

    @property
    def state_preparation(self) -> QuantumCircuit:
        r"""Get the :math:`\mathcal{A}` operator encoding the amplitude :math:`a`.

        Returns:
            The :math:`\mathcal{A}` operator as `QuantumCircuit`.
        """
        return self._state_preparation

    @state_preparation.setter
    def state_preparation(self, state_preparation: QuantumCircuit) -> None:
        r"""Set the :math:`\mathcal{A}` operator, that encodes the amplitude to be estimated.

        Args:
            state_preparation: The new :math:`\mathcal{A}` operator.
        """
        self._state_preparation = state_preparation

    @property
    def post_processing(self) -> Callable[[float], float]:
        """Apply post processing to the input value.

        Returns:
            A handle to the post processing function. Acts as identity by default.
        """
        if self._post_processing is None:
            return lambda x: x

        return self._post_processing

    @post_processing.setter
    def post_processing(self, post_processing: Callable[[float], float]) -> None:
        """Set the post processing function.

        Args:
            post_processing: A handle to the post processing function.
        """
        self._post_processing = post_processing

    @property
    def is_good_state(self) -> Callable[[str], float]:
        """Checks whether a bitstring represents a good state.

        Returns:
            Handle to the ``is_good_state`` callable.
        """
        if self._is_good_state is None:
            return lambda x: all(bit == '1' for bit in x)
            # if self.objective_qubits is None:
            #     raise ValueError('is_good_state can only be called if objective_qubits is set.')

            # return lambda x: all(x[objective] == '1' for objective in self.objective_qubits)

        return self._is_good_state

    @is_good_state.setter
    def is_good_state(self, is_good_state: Callable[[str], float]) -> None:
        """Set the ``is_good_state`` function.

        Args:
            is_good_state: A function to determine whether a bitstring represents a good state.
        """
        self._is_good_state = is_good_state

    @property
    def grover_operator(self) -> Optional[QuantumCircuit]:
        r"""Get the :math:`\mathcal{Q}` operator, or Grover operator.

        If the Grover operator is not set, we try to build it from the :math:`\mathcal{A}` operator
        and `objective_qubits`. This only works if `objective_qubits` is a list of integers.

        Returns:
            The Grover operator, or None if neither the Grover operator nor the
            :math:`\mathcal{A}` operator is  set.
        """
        if self._grover_operator is not None:
            return self._grover_operator

        if self.state_preparation is not None and isinstance(self.objective_qubits, list):
            # build the reflection about the bad state: a MCZ with open controls (thus X gates
            # around the controls) and X gates around the target to change from a phaseflip on
            # |1> to a phaseflip on |0>
            num_state_qubits = self.state_preparation.num_qubits \
                - self.state_preparation.num_ancillas

            oracle = QuantumCircuit(num_state_qubits)
            oracle.x(self.objective_qubits)
            oracle.h(self.objective_qubits[-1])
            if len(self.objective_qubits) == 1:
                oracle.x(self.objective_qubits[0])
            else:
                oracle.mcx(self.objective_qubits[:-1], self.objective_qubits[-1])
            oracle.h(self.objective_qubits[-1])
            oracle.x(self.objective_qubits)

            # construct the grover operator
            return GroverOperator(oracle, self.state_preparation)

        return None

    @grover_operator.setter
    def grover_operator(self, grover_operator: QuantumCircuit) -> None:
        r"""Set the :math:`\mathcal{Q}` operator.

        Args:
            grover_operator: The new :math:`\mathcal{Q}` operator.
        """
        self._grover_operator = grover_operator

    @property
    def objective_qubits(self) -> Optional[List[int]]:
        """Get the criterion for a measurement outcome to be in a 'good' state.

        Returns:
            The criterion as list of qubit indices.
        """
        return self._objective_qubits

    @objective_qubits.setter
    def objective_qubits(self, objective_qubits: List[int]) -> None:
        """Set the criterion for a measurement outcome to be in a 'good' state.

        Args:
            objective_qubits: The criterion as callable of list of qubit indices.
        """
        self._objective_qubits = objective_qubits
