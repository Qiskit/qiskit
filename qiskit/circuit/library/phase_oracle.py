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

"""Phase Oracle object."""

from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.classicalfunction.boolean_expression import BooleanExpression


class PhaseOracle(QuantumCircuit):
    """Phase Oracle."""

    def __init__(self, expression: str) -> None:
        self.bit_oracle = BooleanExpression(expression)
        self.state_qubits = range(self.bit_oracle.num_qubits - 1)   # input qubits for the oracle

        # initialize the quantumcircuit
        qr_state = QuantumRegister(len(self.state_qubits), 'state')

        super().__init__(qr_state, name='Phase Oracle')

        from tweedledum.passes import pkrm_synth  # pylint: disable=no-name-in-module
        synthesizer = lambda logic_network: pkrm_synth(logic_network,
                                                       {"pkrm_synth": {"phase_esop": True}})

        self.compose(self.bit_oracle.synth(synthesizer=synthesizer), inplace=True)

    def evaluate_bitstring(self, bitstring: str) -> bool:
        """Evaluate the oracle on a bitstring.
        This evaluation is done classically without any quantum circuit.
        Args:
            bitstring: The bitstring for which to evaluate.
        Returns:
            True if the bitstring is a good state, False otherwise.
        """
        return self.bit_oracle.simulate(bitstring)
