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


"""XOR circuit."""

from typing import Optional

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.exceptions import CircuitError


class XOR(QuantumCircuit):
    """An n_qubit circuit for bitwise xor-ing the input with some integer ``amount``.

    The ``amount`` is xor-ed in bitstring form with the input.

    This circuit can also represent addition by ``amount`` over the finite field GF(2).
    """

    def __init__(
        self,
        num_qubits: int,
        amount: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Return a circuit implementing bitwise xor.

        Args:
            num_qubits: the width of circuit.
            amount: the xor amount in decimal form.
            seed: random seed in case a random xor is requested.

        Raises:
            CircuitError: if the xor bitstring exceeds available qubits.

        Reference Circuit:
            .. jupyter-execute::
                :hide-code:

                from qiskit.circuit.library import XOR
                import qiskit.tools.jupyter
                circuit = XOR(5, seed=42)
                %circuit_library_info circuit
        """
        circuit = QuantumCircuit(num_qubits, name="xor")

        if amount is not None:
            if len(bin(amount)[2:]) > num_qubits:
                raise CircuitError("Bits in 'amount' exceed circuit width")
        else:
            rng = np.random.default_rng(seed)
            amount = rng.integers(0, 2 ** num_qubits)

        for i in range(num_qubits):
            bit = amount & 1
            amount = amount >> 1
            if bit == 1:
                circuit.x(i)

        super().__init__(*circuit.qregs, name="xor")
        self.compose(circuit.to_gate(), qubits=self.qubits, inplace=True)
