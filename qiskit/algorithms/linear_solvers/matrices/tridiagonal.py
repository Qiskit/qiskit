# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Hamiltonian simulation of tridiagonal Toeplitz symmetric matrices."""
import numpy as np
from scipy.sparse import diags

from qiskit.circuit import QuantumCircuit, QuantumRegister, AncillaRegister
from .linear_system_matrix import LinearSystemMatrix


class Tridiagonal(LinearSystemMatrix):
    """Class of tridiagonal Toeplitz symmetric matrices"""

    def __init__(self, num_state_qubits: int, main_diag: float, off_diag: float,
                 tolerance: float = 1e-2, evo_time: float = 1.0, trotter: int = 1,
                 name: str = 'tridi') -> None:
        """
        Args:
            num_state_qubits: the number of qubits where the unitary acts.
            main_diag: the main diagonal entry
            off_diag: the off diagonal entry
            tolerance: the accuracy desired for the approximation
            evo_time: the time of the Hamiltonian simulation
            trotter: the number of Trotter steps
            name: The name of the object.
        """
        # define internal parameters
        self._num_state_qubits = None
        self._main_entry = None
        self._off_diag = None
        self._tolerance = None
        self._evo_time = None  # makes sure the eigenvalues are contained in [0,1)
        self._trotter = None

        # store parameters
        self.main_entry = main_diag
        self.off_diag = off_diag
        super().__init__(num_state_qubits=num_state_qubits, tolerance=tolerance, evo_time=evo_time,
                         name=name)
        self.trotter = trotter

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
    def main_entry(self) -> float:
        """Return the entry in the main diagonal."""
        return self._main_entry

    @main_entry.setter
    def main_entry(self, main_entry: float) -> None:
        """Set the entry in the main diagonal.
        Args:
            main_entry: The new entry in the main diagonal.
        """
        self._main_entry = main_entry

    @property
    def off_diag(self) -> float:
        """Return the entry in the off diagonals."""
        return self._off_diag

    @off_diag.setter
    def off_diag(self, off_diag: float) -> None:
        """Set the entry in the off diagonals.
        Args:
            off_diag: The new entry in the main diagonal.
        """
        self._off_diag = off_diag

    @property
    def tolerance(self) -> float:
        """Return the error tolerance"""
        return self._tolerance

    @tolerance.setter
    def tolerance(self, tolerance: float) -> None:
        """Set the error tolerance.
        Args:
            tolerance: The new error tolerance.
        """
        self._tolerance = tolerance

    @property
    def evo_time(self) -> float:
        """Return the time of the evolution."""
        return self._evo_time

    @evo_time.setter
    def evo_time(self, evo_time: float) -> None:
        """Set the time of the evolution and update the number of Trotter steps because the error
         tolerance is a function of the evolution time and the number of trotter steps.

        Args:
            evo_time: The new time of the evolution.
        """
        self._evo_time = evo_time
        # Update the number of trotter steps. Max 7 for now, upper bounds too loose.
        self.trotter = int(np.ceil(np.sqrt(((evo_time * np.abs(self.off_diag)) ** 3) / 2
                                           / self.tolerance)))

    @property
    def trotter(self) -> int:
        """Return the number of trotter steps."""
        return self._trotter

    @trotter.setter
    def trotter(self, trotter: int) -> None:
        """Set the number of trotter steps.
        Args:
            trotter: The new number of trotter steps.
        """
        self._trotter = trotter

    @property
    def matrix(self) -> np.ndarray:
        """Return the matrix."""
        matrix = diags([self.off_diag, self.main_entry, self.off_diag], [-1, 0, 1],
                       shape=(2 ** self.num_state_qubits, 2 ** self.num_state_qubits)).toarray()
        return matrix

    @property
    def eigs_bounds(self) -> [float, float]:
        """Return lower and upper bounds on the eigenvalues of the matrix."""
        matrix_array = self.matrix
        lambda_max = max(np.abs(np.linalg.eigvals(matrix_array)))
        lambda_min = min(np.abs(np.linalg.eigvals(matrix_array)))
        return [lambda_min, lambda_max]

    @property
    def condition_bounds(self) -> [float, float]:
        """Return lower and upper bounds on the condition number of the matrix."""
        matrix_array = self.matrix
        kappa = np.linalg.cond(matrix_array)
        return [kappa, kappa]

    def _check_configuration(self, raise_on_failure: bool = True) -> bool:
        valid = True

        if self.trotter < 1:
            valid = False
            if raise_on_failure:
                raise AttributeError('The number of trotter steps should be a positive integer.')

        return valid

    def _reset_registers(self, num_state_qubits: int) -> None:
        """Reset the quantum registers.

        Args:
            num_state_qubits: The number of qubits to represent the matrix.
        """
        qr_state = QuantumRegister(num_state_qubits, 'state')
        self.qregs = [qr_state]
        self._ancillas = []
        self._qubits = qr_state[:]

        if num_state_qubits > 1:
            qr_ancilla = AncillaRegister(max(1, num_state_qubits - 1))
            self.add_register(qr_ancilla)

    def _build(self) -> None:
        """Build the circuit"""
        # do not build the circuit if _data is already populated
        if self._data is not None:
            return

        self._data = []

        # check whether the configuration is valid
        self._check_configuration()

        self.compose(self.power(1), inplace=True)

    def _cn_gate(self, controls: int, ancilla: int, phi: float, ulambda: float, theta: float) \
            -> QuantumCircuit:
        """Apply an n-controlled gate.

        Args:
            controls: number of control qubits
            ancilla: number of ancilla qubits
            phi: argument for a general qiskit u gate
            ulambda: argument for a general qiskit u gate
            theta: argument for a general qiskit u gate

        Returns:
            The quantum circuit implementing a multi-controlled unitary applied as [q_target] +
            qr_controls[:] + qr_ancilla[:].
        """
        qr = QuantumRegister(controls + 1)
        qr_ancilla = QuantumRegister(ancilla)
        qc = QuantumCircuit(qr, qr_ancilla, name='cn_gate')
        q_tgt = qr[0]
        qr_controls = qr[1:]
        # The first Toffoli
        qc.ccx(qr_controls[0], qr_controls[1], qr_ancilla[0])
        for i in range(2, controls):
            qc.ccx(qr_controls[i], qr_ancilla[i - 2], qr_ancilla[i - 1])
        # Now apply the 1-controlled version of the gate with control the last ancilla bit
        qc.cu(theta, phi, ulambda, 0, qr_ancilla[controls - 2], q_tgt)

        # Uncompute ancillae
        for i in range(controls - 1, 1, -1):
            qc.ccx(qr_controls[i], qr_ancilla[i - 2], qr_ancilla[i - 1])
        qc.ccx(qr_controls[0], qr_controls[1], qr_ancilla[0])
        return qc

    def _main_diag(self, theta: float = 1) -> QuantumCircuit:
        """Circuit implementing the matrix consisting of entries in the main diagonal.

        Args:
            theta: Scale factor for the main diagonal entries (e.g. evo_time/trotter_steps).

        Returns:
            The quantum circuit implementing the matrix consisting of entries in the main diagonal.
        """
        theta *= self.main_entry
        qc = QuantumCircuit(self.num_state_qubits, name='main_diag')
        qc.x(0)
        qc.p(theta, 0)
        qc.x(0)
        qc.p(theta, 0)

        def control():
            qc_control = QuantumCircuit(self.num_state_qubits + 1, name='main_diag')
            qc_control.p(theta, 0)
            return qc_control

        qc.control = control
        return qc

    def _off_diags(self, theta: float = 1) -> QuantumCircuit:
        """Circuit implementing the matrix consisting of entries in the off diagonals.

        Args:
            theta: Scale factor for the off diagonal entries (e.g. evo_time/trotter_steps).

        Returns:
            The quantum circuit implementing the matrix consisting of entries in the off diagonals.
        """
        theta *= self.off_diag

        qr = QuantumRegister(self.num_state_qubits)
        if self.num_state_qubits > 1:
            qr_ancilla = AncillaRegister(max(1, self.num_state_qubits - 2))
            qc = QuantumCircuit(qr, qr_ancilla, name='off_diags')
        else:
            qc = QuantumCircuit(qr, name='off_diags')
            qr_ancilla = None

        # Gates for H2 with t
        qc.u(-2 * theta, 3 * np.pi / 2, np.pi / 2, qr[0])

        # Gates for H3
        for i in range(0, self.num_state_qubits - 1):
            q_controls = []
            qc.cx(qr[i], qr[i + 1])
            q_controls.append(qr[i + 1])

            # Now we want controlled by 0
            qc.x(qr[i])
            for j in range(i, 0, -1):
                qc.cx(qr[i], qr[j - 1])
                q_controls.append(qr[j - 1])
            qc.x(qr[i])

            # Multicontrolled x rotation
            if len(q_controls) > 1:
                qc.append(self._cn_gate(len(q_controls), len(qr_ancilla), 3 * np.pi / 2, np.pi / 2,
                                        -2 * theta), [qr[i]] + q_controls[:] + qr_ancilla[:])
            else:
                qc.cu(-2 * theta, 3 * np.pi / 2, np.pi / 2, 0, q_controls[0], qr[i])

            # Uncompute
            qc.x(qr[i])
            for j in range(0, i):
                qc.cx(qr[i], qr[j])
            qc.x(qr[i])
            qc.cx(qr[i], qr[i + 1])

        def control():
            qr_state = QuantumRegister(self.num_state_qubits + 1)
            if self.num_state_qubits > 1:
                qr_ancilla = AncillaRegister(max(1, self.num_state_qubits - 1))
                qc_control = QuantumCircuit(qr_state, qr_ancilla, name='off_diags')
            else:
                qc_control = QuantumCircuit(qr_state, name='off_diags')
                qr_ancilla = None
            # Control will be qr[0]
            q_control = qr_state[0]
            qr = qr_state[1:]
            # Gates for H2 with t
            qc_control.cu(-2 * theta, 3 * np.pi / 2, np.pi / 2, 0, q_control, qr[0])

            # Gates for H3
            for i in range(0, self.num_state_qubits - 1):
                q_controls = []
                q_controls.append(q_control)
                qc_control.cx(qr[i], qr[i + 1])
                q_controls.append(qr[i + 1])

                # Now we want controlled by 0
                qc_control.x(qr[i])
                for j in range(i, 0, -1):
                    qc_control.cx(qr[i], qr[j - 1])
                    q_controls.append(qr[j - 1])
                qc_control.x(qr[i])

                # Multicontrolled x rotation
                if len(q_controls) > 1:
                    qc_control.append(self._cn_gate(len(q_controls), len(qr_ancilla),
                                                    3 * np.pi / 2, np.pi / 2, -2 * theta),
                                      [qr[i]] + q_controls[:] + qr_ancilla[:])
                else:
                    qc_control.cu(-2 * theta, 3 * np.pi / 2, np.pi / 2, 0, q_controls[0], qr[i])

                # Uncompute
                qc_control.x(qr[i])
                for j in range(0, i):
                    qc_control.cx(qr[i], qr[j])
                qc_control.x(qr[i])
                qc_control.cx(qr[i], qr[i + 1])
            return qc_control

        qc.control = control
        return qc

    def inverse(self):
        return Tridiagonal(self.num_state_qubits, self.main_entry, self.off_diag,
                           evo_time=-1 * self.evo_time)

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
        qc_raw = QuantumCircuit(self.num_state_qubits)

        def control():
            qr_state = QuantumRegister(self.num_state_qubits + 1, 'state')
            if self.num_state_qubits > 1:
                qr_ancilla = AncillaRegister(max(1, self.num_state_qubits - 1))
                qc = QuantumCircuit(qr_state, qr_ancilla, name='exp(iHk)')
            else:
                qc = QuantumCircuit(qr_state, name='exp(iHk)')
                qr_ancilla = None
            # Control will be qr[0]
            q_control = qr_state[0]
            qr = qr_state[1:]
            # Since A1 commutes, one application with evo_time*2^{j} to the last qubit is enough
            qc.append(self._main_diag(self.evo_time * power).control(), [q_control] + qr[:])

            # Update trotter step to compensate the error
            trotter_new = int(np.ceil(np.sqrt(power) * self.trotter))

            # exp(iA2t/2m)
            qc.u(self.off_diag * self.evo_time * power / trotter_new, 3 * np.pi / 2, np.pi / 2,
                 qr[0])
            # for _ in range(power):
            for _ in range(0, trotter_new):
                if qr_ancilla:
                    qc.append(self._off_diags(self.evo_time * power / trotter_new).control(),
                              [q_control] + qr[:] + qr_ancilla[:])
                else:
                    qc.append(self._off_diags(self.evo_time * power / trotter_new).control(),
                              [q_control] + qr[:])
            # exp(-iA2t/2m)
            qc.u(-self.off_diag * self.evo_time * power / trotter_new, 3 * np.pi / 2, np.pi / 2,
                 qr[0])
            return qc

        qc_raw.control = control
        return qc_raw
