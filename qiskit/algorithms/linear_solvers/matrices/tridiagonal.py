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

from typing import Optional

import numpy as np
from scipy.sparse import diags

from qiskit.circuit import QuantumCircuit, QuantumRegister, AncillaRegister


class Tridiagonal(QuantumCircuit):
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
        """

        qr_state = QuantumRegister(num_state_qubits, 'state')
        if num_state_qubits > 1:
            qr_ancilla = AncillaRegister(max(1, num_state_qubits - 1))
            super().__init__(qr_state, qr_ancilla, name=name)
        else:
            super().__init__(qr_state, name=name)

        self._num_state_qubits = None

        self._main_entry = main_diag
        self._off_diag = off_diag
        self._tolerance = tolerance
        self._evo_time = evo_time
        self._trotter = trotter

        self._num_state_qubits = num_state_qubits
        self.compose(self.power(1), inplace=True)

    @property
    def tolerance(self) -> float:
        """Return the error tolerance"""
        return self._tolerance

    @tolerance.setter
    def tolerance(self, tolerance: float):
        """Set the error tolerance"""
        self._tolerance = tolerance

    @property
    def evo_time(self) -> float:
        """Return the time of the evolution"""
        return self._evo_time

    @evo_time.setter
    def evo_time(self, evo_time: float):
        """Set the time of the evolution and update the number of Trotter steps because the error
         tolerance is a function of the evolution time and the number of trotter steps"""
        self._evo_time = evo_time
        # Update the number of trotter steps. Max 7 for now, upper bounds too loose.
        self._trotter = min(self._num_state_qubits + 1,
                            int(np.ceil(np.sqrt(((evo_time * np.abs(self._off_diag)) ** 3)
                                                / 2 / self._tolerance))))

    def matrix(self) -> np.ndarray:
        """Return the matrix"""
        matrix = diags([self._off_diag, self._main_entry, self._off_diag], [-1, 0, 1],
                       shape=(2 ** self._num_state_qubits, 2 ** self._num_state_qubits)).toarray()
        return matrix

    def _cn_gate(self, controls: int, ancilla: int, phi: float, ulambda: float, theta: float)\
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
        qc = QuantumCircuit(qr, qr_ancilla)
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
        theta *= self._main_entry
        qc = QuantumCircuit(self._num_state_qubits)
        qc.x(0)
        qc.p(theta, 0)
        qc.x(0)
        qc.p(theta, 0)

        def control():
            qc_control = QuantumCircuit(self._num_state_qubits + 1)
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
        theta *= self._off_diag

        qr = QuantumRegister(self._num_state_qubits)
        if self._num_state_qubits > 1:
            qr_ancilla = AncillaRegister(max(1, self._num_state_qubits - 2))
            qc = QuantumCircuit(qr, qr_ancilla)
        else:
            qc = QuantumCircuit(qr)
            qr_ancilla = None

        # Gates for H2 with t
        qc.u(-2 * theta, 3 * np.pi / 2, np.pi / 2, qr[0])

        # Gates for H3
        for i in range(0, self._num_state_qubits - 1):
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
            qr_state = QuantumRegister(self._num_state_qubits + 1)
            if self._num_state_qubits > 1:
                qr_ancilla = AncillaRegister(max(1, self._num_state_qubits - 1))
                qc_control = QuantumCircuit(qr_state, qr_ancilla)
            else:
                qc_control = QuantumCircuit(qr_state)
                qr_ancilla = None
            # Control will be qr[0]
            q_control = qr_state[0]
            qr = qr_state[1:]
            # Gates for H2 with t
            qc_control.cu(-2 * theta, 3 * np.pi / 2, np.pi / 2, 0, q_control, qr[0])

            # Gates for H3
            for i in range(0, self._num_state_qubits - 1):
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
        return Tridiagonal(self._num_state_qubits, self._main_entry, self._off_diag,
                           evo_time=-1 * self._evo_time)

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
        qc_raw = QuantumCircuit(self._num_state_qubits)

        def control():
            qr_state = QuantumRegister(self._num_state_qubits + 1, 'state')
            if self._num_state_qubits > 1:
                qr_ancilla = AncillaRegister(max(1, self._num_state_qubits - 1))
                qc = QuantumCircuit(qr_state, qr_ancilla)
            else:
                qc = QuantumCircuit(qr_state)
                qr_ancilla = None
            # Control will be qr[0]
            q_control = qr_state[0]
            qr = qr_state[1:]
            # Since A1 commutes, one application with evo_time*2^{j} to the last qubit is enough
            qc.append(self._main_diag(self._evo_time * power).control(), [q_control] + qr[:])

            # Update trotter step to compensate the error
            trotter_new = int(np.ceil(np.sqrt(power) * self._trotter))

            # exp(iA2t/2m)
            qc.u(self._off_diag * self._evo_time * power / trotter_new, 3 * np.pi / 2, np.pi / 2,
                 qr[0])
            # for _ in range(power):
            for _ in range(0, trotter_new):
                if qr_ancilla:
                    qc.append(self._off_diags(self._evo_time * power / trotter_new).control(),
                              [q_control] + qr[:] + qr_ancilla[:])
                else:
                    qc.append(self._off_diags(self._evo_time * power / trotter_new).control(),
                              [q_control] + qr[:])
            # exp(-iA2t/2m)
            qc.u(-self._off_diag * self._evo_time * power / trotter_new, 3 * np.pi / 2, np.pi / 2,
                 qr[0])
            return qc

        qc_raw.control = control
        return qc_raw
