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
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import GroverOperator
from qiskit.quantum_info import Statevector


class AmplificationProblem:
    """The amplification problem is the input to amplitude amplification algorithms, like Grover.

    This class contains all problem-specific information required to run an amplitude amplification
    algorithm. It minimally contains the Grover operator. It can further hold some post processing
    on the optimal bitstring.
    """

    def __init__(
        self,
        oracle: QuantumCircuit | Statevector,
        state_preparation: QuantumCircuit | None = None,
        grover_operator: QuantumCircuit | None = None,
        post_processing: Callable[[str], Any] | None = None,
        objective_qubits: int | list[int] | None = None,
        is_good_state: Callable[[str], bool] | list[int] | list[str] | Statevector | None = None,
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
            is_good_state: A function to check whether a string represents a good state. By default
                if the ``oracle`` argument has an ``evaluate_bitstring`` method (currently only
                provided by the :class:`~qiskit.circuit.library.PhaseOracle` class) this will be
                used, otherwise this kwarg is required and **must** be specified.
        """
        self._oracle = oracle
        self._state_preparation = state_preparation
        self._grover_operator = grover_operator
        self._post_processing = post_processing
        self._objective_qubits = objective_qubits
        if is_good_state is not None:
            self._is_good_state = is_good_state
        elif hasattr(oracle, "evaluate_bitstring"):
            self._is_good_state = oracle.evaluate_bitstring
        else:
            self._is_good_state = None

    @property
    def oracle(self) -> QuantumCircuit | Statevector:
        """Return the oracle.

        Returns:
            The oracle.
        """
        return self._oracle

    @oracle.setter
    def oracle(self, oracle: QuantumCircuit | Statevector) -> None:
        """Set the oracle.

        Args:
            oracle: The oracle.
        """
        self._oracle = oracle

    @property
    def state_preparation(self) -> QuantumCircuit:
        r"""Get the state preparation operator :math:`\mathcal{A}`.

        Returns:
            The :math:`\mathcal{A}` operator as `QuantumCircuit`.
        """
        if self._state_preparation is None:
            state_preparation = QuantumCircuit(self.oracle.num_qubits)
            state_preparation.h(state_preparation.qubits)
            return state_preparation

        return self._state_preparation

    @state_preparation.setter
    def state_preparation(self, state_preparation: QuantumCircuit | None) -> None:
        r"""Set the :math:`\mathcal{A}` operator. If None, a layer of Hadamard gates is used.

        Args:
            state_preparation: The new :math:`\mathcal{A}` operator or None.
        """
        self._state_preparation = state_preparation

    @property
    def post_processing(self) -> Callable[[str], Any]:
        """Apply post processing to the input value.

        Returns:
            A handle to the post processing function. Acts as identity by default.
        """
        if self._post_processing is None:
            return lambda x: x

        return self._post_processing

    @post_processing.setter
    def post_processing(self, post_processing: Callable[[str], Any]) -> None:
        """Set the post processing function.

        Args:
            post_processing: A handle to the post processing function.
        """
        self._post_processing = post_processing

    @property
    def objective_qubits(self) -> list[int]:
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
    def objective_qubits(self, objective_qubits: int | list[int] | None) -> None:
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
        if (self._is_good_state is None) or callable(self._is_good_state):
            return self._is_good_state  # returns None if no is_good_state arg has been set
        elif isinstance(self._is_good_state, list):
            if all(isinstance(good_bitstr, str) for good_bitstr in self._is_good_state):
                return lambda bitstr: bitstr in self._is_good_state
            else:
                return lambda bitstr: all(
                    bitstr[good_index] == "1" for good_index in self._is_good_state
                )

        return lambda bitstr: bitstr in self._is_good_state.probabilities_dict()

    @is_good_state.setter
    def is_good_state(
        self, is_good_state: Callable[[str], bool] | list[int] | list[str] | Statevector
    ) -> None:
        """Set the ``is_good_state`` function.

        Args:
            is_good_state: A function to determine whether a bitstring represents a good state.
        """
        self._is_good_state = is_good_state

    @property
    def grover_operator(self) -> QuantumCircuit | None:
        r"""Get the :math:`\mathcal{Q}` operator, or Grover operator.

        If the Grover operator is not set, we try to build it from the :math:`\mathcal{A}` operator
        and `objective_qubits`. This only works if `objective_qubits` is a list of integers.

        Returns:
            The Grover operator, or None if neither the Grover operator nor the
            :math:`\mathcal{A}` operator is  set.
        """
        if self._grover_operator is None:
            return GroverOperator(self.oracle, self.state_preparation)
        return self._grover_operator

    @grover_operator.setter
    def grover_operator(self, grover_operator: QuantumCircuit | None) -> None:
        r"""Set the :math:`\mathcal{Q}` operator.

        If None, this operator is constructed from the ``oracle`` and ``state_preparation``.

        Args:
            grover_operator: The new :math:`\mathcal{Q}` operator or None.
        """
        self._grover_operator = grover_operator
