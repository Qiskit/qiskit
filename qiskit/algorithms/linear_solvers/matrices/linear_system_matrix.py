# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""An abstract class for matrices input to the linear systems solvers in Qiskit."""

from abc import ABC, abstractmethod
from typing import Tuple

from qiskit import QuantumCircuit
from qiskit.circuit.library import BlueprintCircuit


class LinearSystemMatrix(BlueprintCircuit, ABC):
    """Base class for linear system matrices."""

    def __init__(
        self,
        num_state_qubits: int,
        tolerance: float,
        evolution_time: float,
        name: str = "ls_matrix",
    ) -> None:
        """
        Args:
            num_state_qubits: the number of qubits where the unitary acts.
            tolerance: the accuracy desired for the approximation
            evolution_time: the time of the Hamiltonian simulation
            name: The name of the object.
        """
        super().__init__(name=name)

        # define internal parameters
        self._num_state_qubits = None
        self._tolerance = None
        self._evolution_time = None  # makes sure the eigenvalues are contained in [0,1)

        # store parameters
        self.num_state_qubits = num_state_qubits
        self.tolerance = tolerance
        self.evolution_time = evolution_time

    @property
    def num_state_qubits(self) -> int:
        r"""The number of state qubits representing the state :math:`|x\rangle`.

        Returns:
            The number of state qubits.
        """
        return self._num_state_qubits

    @num_state_qubits.setter
    def num_state_qubits(self, num_state_qubits: int) -> None:
        """Set the number of state qubits.

        Note that this may change the underlying quantum register, if the number of state qubits
        changes.

        Args:
            num_state_qubits: The new number of qubits.
        """
        if num_state_qubits != self._num_state_qubits:
            self._invalidate()
            self._num_state_qubits = num_state_qubits
            self._reset_registers(num_state_qubits)

    @property
    def tolerance(self) -> float:
        """Return the error tolerance"""
        return self._tolerance

    @tolerance.setter
    def tolerance(self, tolerance: float) -> None:
        """Set the error tolerance
        Args:
            tolerance: The new error tolerance.
        """
        self._tolerance = tolerance

    @property
    def evolution_time(self) -> float:
        """Return the time of the evolution."""
        return self._evolution_time

    @evolution_time.setter
    def evolution_time(self, evolution_time: float) -> None:
        """Set the time of the evolution.

        Args:
            evolution_time: The new time of the evolution.
        """
        self._evolution_time = evolution_time

    @abstractmethod
    def eigs_bounds(self) -> Tuple[float, float]:
        """Return lower and upper bounds on the eigenvalues of the matrix."""
        raise NotImplementedError

    @abstractmethod
    def condition_bounds(self) -> Tuple[float, float]:
        """Return lower and upper bounds on the condition number of the matrix."""
        raise NotImplementedError

    @abstractmethod
    def _reset_registers(self, num_state_qubits: int) -> None:
        """Reset the registers according to the new number of state qubits.

        Args:
            num_state_qubits: The new number of qubits.
        """
        raise NotImplementedError

    @abstractmethod
    def power(self, power: int, matrix_power: bool = False) -> QuantumCircuit:
        """Build powers of the circuit.

        Args:
            power: The power to raise this circuit to.
            matrix_power: If True, the circuit is converted to a matrix and then the
                matrix power is computed. If False, and ``power`` is a positive integer,
                the implementation defaults to ``repeat``.

        Returns:
            The quantum circuit implementing powers of the unitary.
        """
        raise NotImplementedError
