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

from qiskit.circuit import BooleanExpression, QuantumCircuit, QuantumRegister


class PhaseOracle(QuantumCircuit):
    """Phase Oracle."""

    def __init__(self, expression: str) -> None:
        self.bit_oracle = BooleanExpression(expression)
        self.state_qubits = range(self.bit_oracle.num_qubits - 1)   # input qubits for the oracle

        # initialize the quantumcircuit
        qr_state = QuantumRegister(len(self.state_qubits), 'state')
        qr_flag = QuantumRegister(1, 'flag')
        super().__init__(qr_state, qr_flag, name='Phase Oracle')

        # to convert from the bitflip oracle provided from BooleanExpression to a phase oracle, we
        # additionally apply a Hadamard and X gates
        self.x(qr_flag)
        self.h(qr_flag)
        self.compose(self.bit_oracle.synth(), inplace=True)
        self.h(qr_flag)
        self.x(qr_flag)

    def evaluate_bitstring(self, bitstring: str) -> bool:
        """Evaluate the oracle on a bitstring.
        This evaluation is done classically without any quantum circuit.
        Args:
            bitstring: The bitstring for which to evaluate.
        Returns:
            True if the bitstring is a good state, False otherwise.
        """
        return self.bit_oracle.simulate(bitstring)
