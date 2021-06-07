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

"""Hamiltonian simulation of matrices given as numpy arrays."""

from typing import Tuple
import numpy as np
import scipy as sp

from qiskit import QuantumCircuit, QuantumRegister

from .linear_system_matrix import LinearSystemMatrix


class NumPyMatrix(LinearSystemMatrix):
    """Class of matrices given as a numpy array.

    Examples:

        .. jupyter-execute::

            import numpy as np
            from qiskit import QuantumCircuit
            from qiskit.algorithms.linear_solvers.matrices.numpy_matrix import NumPyMatrix

            matrix = NumPyMatrix(np.array([[1 / 2, 1 / 6, 0, 0], [1 / 6, 1 / 2, 1 / 6, 0],
                               [0, 1 / 6, 1 / 2, 1 / 6], [0, 0, 1 / 6, 1 / 2]]))
            power = 2

            num_qubits = matrix.num_state_qubits
            # Controlled power (as used within QPE)
            pow_circ = matrix.power(power).control()
            circ_qubits = pow_circ.num_qubits
            qc = QuantumCircuit(circ_qubits)
            qc.append(matrix.power(power).control(), list(range(circ_qubits)))
    """

    def __init__(
        self,
        matrix: np.ndarray,
        tolerance: float = 1e-2,
        evolution_time: float = 1.0,
        name: str = "np_matrix",
    ) -> None:
        """
        Args:
            matrix: The matrix defining the linear system problem.
            tolerance: The accuracy desired for the approximation.
            evolution_time: The time of the Hamiltonian simulation.
            name: The name of the object.
        """

        # define internal parameters
        self._num_state_qubits = None
        self._tolerance = None
        self._evolution_time = None  # makes sure the eigenvalues are contained in [0,1)
        self._matrix = None

        # store parameters
        self.num_state_qubits = int(np.log2(matrix.shape[0]))
        self.tolerance = tolerance
        self.evolution_time = evolution_time
        self.matrix = matrix
        super().__init__(
            num_state_qubits=int(np.log2(matrix.shape[0])),
            tolerance=tolerance,
            evolution_time=evolution_time,
            name=name,
        )

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

    @property
    def matrix(self) -> np.ndarray:
        """Return the matrix."""
        return self._matrix

    @matrix.setter
    def matrix(self, matrix: np.ndarray) -> None:
        """Set the matrix.

        Args:
            matrix: The new matrix.
        """
        self._matrix = matrix

    def eigs_bounds(self) -> Tuple[float, float]:
        """Return lower and upper bounds on the eigenvalues of the matrix."""
        matrix_array = self.matrix
        lambda_max = max(np.abs(np.linalg.eigvals(matrix_array)))
        lambda_min = min(np.abs(np.linalg.eigvals(matrix_array)))
        return lambda_min, lambda_max

    def condition_bounds(self) -> Tuple[float, float]:
        """Return lower and upper bounds on the condition number of the matrix."""
        matrix_array = self.matrix
        kappa = np.linalg.cond(matrix_array)
        return kappa, kappa

    def _check_configuration(self, raise_on_failure: bool = True) -> bool:
        valid = True

        if self.matrix.shape[0] != self.matrix.shape[1]:
            if raise_on_failure:
                raise AttributeError("Input matrix must be square!")
            return False
        if np.log2(self.matrix.shape[0]) % 1 != 0:
            if raise_on_failure:
                raise AttributeError("Input matrix dimension must be 2^n!")
            return False
        if not np.allclose(self.matrix, self.matrix.conj().T):
            if raise_on_failure:
                raise AttributeError("Input matrix must be hermitian!")
            return False

        return valid

    def _reset_registers(self, num_state_qubits: int) -> None:
        """Reset the quantum registers.

        Args:
            num_state_qubits: The number of qubits to represent the matrix.
        """
        qr_state = QuantumRegister(num_state_qubits, "state")
        self.qregs = [qr_state]
        self._qubits = qr_state[:]

    def _build(self) -> None:
        """Build the circuit"""
        # do not build the circuit if _data is already populated
        if self._data is not None:
            return

        self._data = []

        # check whether the configuration is valid
        self._check_configuration()

        self.compose(self.power(1), inplace=True)

    def inverse(self):
        return NumPyMatrix(self.matrix, evolution_time=-1 * self.evolution_time)

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
        qc = QuantumCircuit(self.num_state_qubits)
        evolved = sp.linalg.expm(1j * self.matrix * self.evolution_time)
        qc.unitary(evolved, qc.qubits)
        return qc.power(power)
