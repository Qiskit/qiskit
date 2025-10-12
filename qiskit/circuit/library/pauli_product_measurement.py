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

from __future__ import annotations

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

    def __init__(
        self,
        pauli: Pauli,
        label: str | None = None,
    ):
        """
        Args:
            pauli: A tensor product of Pauli operators defining the measurement,
                for example ``Pauli("XY")`` or ``Pauli("-XYIZ)``.
                The identity Pauli operator is not permitted.
                The Pauli may include a phase of :math:`-1`, but not :math:`i` or :math:`-i`.
            label: An optional label for the gate to display in circuit visualizations.
                By default, the label is set to ``PPM(<pauli label>)``.

        .. note::

            While Paulis involving "I"-terms are fully supported, it is recommended to remove
            "I"-terms from the Pauli when creating a ``PauliProductMeasurement`` instruction,
            as this does not change the actual measurement but specifies the instruction over
            a smaller set of qubits.

        """

        if not isinstance(pauli, Pauli):
            raise CircuitError(
                "A Pauli Product Measurement instruction can only be "
                "instantiated from a Pauli object."
            )

        if _is_identity_label(pauli):
            raise CircuitError(
                "A Pauli Product Measurement instruction can not have an all-'I' label."
            )

        if pauli.phase not in [0, 2]:
            raise CircuitError(
                "A Pauli Product Measurement instruction can not have a Pauli phase of i or -i."
            )

        num_qubits = len(pauli.z)
        params = [pauli.z, pauli.x, pauli.phase]

        if label is None:
            label = _get_default_label(pauli)

        super().__init__(
            name="PauliProductMeasurement",
            num_qubits=num_qubits,
            num_clbits=1,
            params=params,
            label=label,
        )

    @classmethod
    def _from_pauli_data(cls, z, x, phase, label):
        """
        Instantiates a PauliProductMeasurement isntruction from pauli data and label.
        This function is used internally from within the rust code.
        """
        return cls(Pauli((z, x, phase)), label)

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


def _get_default_label(pauli: Pauli):
    """Creates the detault label for PauliProductMeasurement instruction,
    used for visualization.
    """
    return "PPM(" + pauli.to_label() + ")"


def _is_identity_label(pauli: Pauli):
    """Return whether a Pauli has an all-'I' label (up to a phase)."""
    return not np.any(pauli.z) and not np.any(pauli.x)
