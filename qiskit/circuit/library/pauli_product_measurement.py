# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""An instruction to implement a Pauli Product Measurement."""

import numpy as np

from qiskit.circuit import QuantumCircuit, CircuitError
from qiskit.circuit.instruction import Instruction
from qiskit.quantum_info.operators.symplectic.pauli import Pauli

from qiskit._accelerate.synthesis.pauli_product_measurement import synth_pauli_product_measurement


class PauliProductMeasurement(Instruction):
    """Pauli Product Measurement instruction.

    A Pauli Product Measurement is a fundamental operation in fault-tolerant quantum
    computing. Mathematically, it corresponds to a joint projective measurement on
    multiple qubits, where the measured observable is a tensor product of Pauli operators.
    The outcome of this measurement is a single eigenvalue, either :math:`+1` or :math:`-1`,
    indicating the eigenstate of the Pauli product.

    References:

    [1] Daniel Litinski.
    "A Game of Surface Codes: Large-Scale Quantum Computing with Lattice Surgery"
    `arXiv:1808.02892 <https://arxiv.org/abs/1808.02892>`__
    """

    # we will accept pauli but internally we will store something else
    def __init__(
        self,
        data: Pauli | tuple,
        label: str | None = None,
    ):
        """
        Args:
            data: Pauli or tuple (z, x, phase).
            label: A label for the gate to display in visualizations. Per default, the label is
                set to ``exp(-it <operators>)`` where ``<operators>`` is the sum of the Paulis.
        """

        if isinstance(data, Pauli):
            print(f"=> Param of type Pauli")
            if data.phase not in [0, 2]:
                raise CircuitError("Pauli phase of i or -i is not acceptable.")
            params = [data.z, data.x, data.phase]
            num_qubits = len(data.z)
        else:
            params = list(data)
            num_qubits = len(data[0])

        if label is None:
            label = _get_default_label(*params)

        print(f"=> {params = }, {num_qubits = }")

        super().__init__(
            name="PauliProductMeasurement",
            num_qubits=num_qubits,
            num_clbits=1,
            params=params,
            label=label,
        )
        print(f"In PPM::constructor: {self.params = }")

    def inverse(self, annotated=False):
        """Prevents from calling ``inverse`` on a PauliProductMeasurement instruction."""
        raise CircuitError("PauliProductMeasurement is not invertible.")

    def __eq__(self, other):
        if not isinstance(other, PauliProductMeasurement):
            return False

        if self.label != other.label:
            return False

        return (
            np.all(self.params[0] == other.params[0])
            and np.all(self.params[1] == other.params[1])
            and (self.params[2] == other.params[2])
        )

    def _define(self):
        circuit = QuantumCircuit._from_circuit_data(
            synth_pauli_product_measurement(self),
            legacy_qubits=True,
            name="ppm_circuit",
        )
        self.definition = circuit


def _get_default_label(z, x, phase):
    pauli_label = Pauli((z, x, phase)).to_label()
    return "PPM(" + pauli_label + ")"
