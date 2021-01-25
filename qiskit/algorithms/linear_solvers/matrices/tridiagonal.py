from typing import Optional

import numpy as np
from scipy.sparse import diags

from qiskit.circuit import QuantumCircuit, QuantumRegister, AncillaRegister, Qubit

class Tridiagonal(QuantumCircuit):
    """Class of tridiagonal Toeplitz symmetric matrices"""

    def __init__(self, num_state_qubits: int, main_diag: float, off_diag: float,
                 tolerance: Optional[float] = None, time: Optional[float] = None,
                 trotter: Optional[int] = 1, name: str = 'tridi') -> None:

        qr_state = QuantumRegister(num_state_qubits)
        if num_state_qubits > 1:
            qr_ancilla = AncillaRegister(max(1, num_state_qubits - 1))
            super().__init__(qr_state, qr_ancilla, name=name)
        else:
            super().__init__(qr_state, name=name)

        self._num_state_qubits = None

        self._main_diag = main_diag
        self._off_diag = off_diag
        self._tolerance = tolerance if tolerance is not None else 1e-2
        self._time = time if time is not None else 1
        self._trotter = trotter

        self._num_state_qubits = num_state_qubits

    def set_simulation_params(self, time: float, tolerance: float):
        self.tolerance(tolerance)
        self.time(time)

    def tolerance(self, tolerance: float):
        self._tolerance = tolerance

    def time(self, time: float):
        self._time = time
        # Update the number of trotter steps. Max 7 for now, upper bounds too loose.
        self._trotter = min(self._num_state_qubits + 1,int(np.ceil(np.sqrt(((time * np.abs(self._off_diag)) ** 3) / 2 /
                                                  self._tolerance))))

    def matrix(self) -> np.ndarray:
        """Return the matrix"""
        matrix = diags([self._off_diag, self._main_diag, self._off_diag], [-1, 0, 1],
                       shape=(2 ** self._num_state_qubits, 2 ** self._num_state_qubits)).toarray()
        return matrix

    def _cn_gate(self, qc: QuantumCircuit, controls: QuantumRegister, qr_a: AncillaRegister,
                 phi: float, ulambda: float, theta: float, tgt: Qubit):
        """Apply an n-controlled gate.

        Args:
            controls: list of controlling qubits
            qr_a: ancilla register
            phi: argument for a general qiskit u gate
            ulambda: argument for a general qiskit u gate
            theta: argument for a general qiskit u gate
            tgt: target qubit
        """
        # The first Toffoli
        qc.ccx(controls[0], controls[1], qr_a[0])
        for i in range(2, len(controls)):
            qc.ccx(controls[i], qr_a[i - 2], qr_a[i - 1])
        # Now apply the 1-controlled version of the gate with control the last ancilla bit
        qc.cu(theta, phi, ulambda, 0, qr_a[len(controls) - 2], tgt)

        # Uncompute ancillae
        for i in range(len(controls) - 1, 1, -1):
            qc.ccx(controls[i], qr_a[i - 2], qr_a[i - 1])
        qc.ccx(controls[0], controls[1], qr_a[0])
        return qc

    # Controlled version of the circuit for the main diagonal
    def _build_main_controlled(self, qc: QuantumCircuit, q_control: Qubit, params: Optional[float] = 1):
        """Controlled circuit for the matrix consisting of entries in the main diagonal.

        Args:
            q_control: The control qubit.
            params: Argument for the rotation.
        """
        qc.p(params, q_control)
        return qc

    # Controlled version of the circuit for the main diagonal
    def _build_off_diag_controlled(self, qc: QuantumCircuit, q_control: Qubit, qr: QuantumRegister,
                                   qr_anc: Optional[AncillaRegister] = None,
                                   params: Optional[float] = 1) -> QuantumCircuit:
        """Controlled circuit for the matrix consisting of entries in the off diagonals.

        Args:
            qc: The quantum circuit.
            q_control: The control qubit.
            qr: The quantum register where the circuit is built.
            qr_anc: The quantum register containing the ancilla qubits.
            params: Argument for the rotation.
        """
        # Gates for H2 with t
        qc.cu(-2 * params, 3 * np.pi / 2, np.pi / 2, 0, q_control, qr[0])

        # Gates for H3
        for i in range(0, self._num_state_qubits - 1):
            q_controls = []
            q_controls.append(q_control)
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
                self._cn_gate(qc, q_controls, qr_anc, 3 * np.pi / 2, np.pi / 2, -2 * params, qr[i])
            else:
                qc.cu(-2 * params, 3 * np.pi / 2, np.pi / 2, 0, q_controls[0], qr[i])

            # Uncompute
            qc.x(qr[i])
            for j in range(0, i):
                qc.cx(qr[i], qr[j])
            qc.x(qr[i])
            qc.cx(qr[i], qr[i + 1])

        return qc

    def inverse(self):
        self._time = - self._time

    def power(self, power: int):
        """Build powers of the circuit.

        Args:
            power: The exponent.
        """
        qc_raw = QuantumCircuit(self._num_state_qubits)

        def control():
            qr_state = QuantumRegister(self._num_state_qubits + 1)
            if self._num_state_qubits > 1:
                qr_ancilla = AncillaRegister(max(1, self._num_state_qubits - 1))
                qc = QuantumCircuit(qr_state, qr_ancilla)
            else:
                qc = QuantumCircuit(qr_state)
                qr_ancilla = None
            # Control will be qr[0]
            q_control = qr_state[0]
            qr = qr_state[1:]
            # Since A1 commutes, one application with time t*2^{j} to the last qubit is enough
            self._build_main_controlled(qc, q_control, self._main_diag * self._time * power)

            # Update trotter step to compensate the error
            trotter_new = int(np.ceil(np.sqrt(power) * self._trotter))

            # exp(iA2t/2m)
            qc.u(self._off_diag * self._time * power / trotter_new, 3 * np.pi / 2, np.pi / 2, qr[0])
            # for _ in range(power):
            for _ in range(0, trotter_new):
                self._build_off_diag_controlled(qc, q_control, qr, qr_ancilla,
                                                self._time * self._off_diag * power / trotter_new)
            # exp(-iA2t/2m)
            qc.u(-self._off_diag * self._time * power / trotter_new, 3 * np.pi / 2, np.pi / 2, qr[0])
            return qc

        qc_raw.control = control
        return qc_raw
