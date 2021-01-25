import numpy as np
from math import isclose
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.extensions.quantum_initializer import UCRYGate


class ExactInverse(QuantumCircuit):
    """Temporal class to test the new hhl algorithm"""

    def __init__(self, num_state_qubits: int, constant: float, name: str = 'inv') -> None:

        qr_state = QuantumRegister(num_state_qubits)
        qr_flag = QuantumRegister(1)
        super().__init__(qr_state, qr_flag, name=name)

        angles = [0.0]
        nl = 2 ** num_state_qubits

        for i in range(1, nl):
            if isclose(constant * nl / i, 1, abs_tol=1e-5):
                angles.append(np.pi)
            elif constant * nl / i < 1:
                angles.append(2 * np.arcsin(constant * nl / i))
            else:
                angles.append(0.0)

        self.compose(UCRYGate(angles), [qr_flag[0]] + qr_state[:], inplace=True)
