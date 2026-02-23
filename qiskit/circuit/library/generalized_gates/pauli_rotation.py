# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A generic Pauli rotation gate."""

from __future__ import annotations

import typing
import numpy as np
import scipy as sc

from qiskit.circuit import QuantumCircuit, CircuitError, Gate, ParameterExpression
from qiskit.circuit.tools import pi_check
from qiskit._accelerate.circuit_library import pauli_evolution

if typing.TYPE_CHECKING:
    from qiskit.quantum_info import Pauli
    from qiskit.circuit.quantumcircuit import ParameterValueType


class PauliRotationGate(Gate):
    r"""A generic Pauli rotation.

    This implements the unitary operation

    .. math::

        R_P(\theta) = e^{-i \theta / 2 P}

    for a Pauli :math:`P \in \{I, X, Y, Z\}^{\otimes n}` and a rotation angle
    :math:`\theta \in \mathbb R`, which could be represented by a unbound parameter.
    """

    def __init__(
        self,
        pauli: Pauli,
        angle: ParameterValueType,
        label: str | None = None,
    ):
        """
        Args:
            pauli: The Pauli defining the rotation axis. May include a phase of :math:`-1`, but
                not :math:`i` or :math:`-i`.
            angle: The rotation angle.
            label: An optional label for the gate to display in circuit visualizations.
        """
        if len(pauli.x) == 0:
            raise CircuitError("A Pauli rotation requires at least one qubit.")

        if pauli.phase not in [0, 2]:
            raise CircuitError("A Pauli rotation can not have a Pauli phase of i or -i.")

        num_qubits = len(pauli.z)
        self._pauli_z = pauli.z
        self._pauli_x = pauli.x
        # Absorb the sign of the Pauli into the rotation angle
        params = [angle if pauli.phase == 0 else -angle]

        if label is None:
            label = f"R_{pauli.to_label()}"

        super().__init__(
            name="pauli_rotation",
            num_qubits=num_qubits,
            params=params,
            label=label,
        )

    @classmethod
    def _from_pauli_data(cls, z, x, angle, label=None):
        """
        Instantiates a PauliRotationGate from raw pauli data, the angle, and the label.
        This function is used internally from within the rust code and from QPY
        serialization.
        """
        from qiskit.quantum_info import Pauli  # pylint: disable=cyclic-import

        return cls(Pauli((z, x, 0)), angle, label)

    def inverse(self, annotated=False):
        """Return the inverse; a rotation about the negative angle."""
        return PauliRotationGate(self.pauli(), -self.params[0])

    def __eq__(self, other):
        if not isinstance(other, PauliRotationGate):
            return False

        return (
            np.all(self._pauli_z == other._pauli_z)
            and np.all(self._pauli_x == other._pauli_x)
            and (self.params[0] == other.params[0])
        )

    def _define(self):
        # In the Pauli tensor order, the label i corresponds to qubit n-(i+1), so we
        # revert the labels. The code below is consistent with PauliEvolutionGate.
        label = self.pauli().to_label()[::-1]
        angle = self.params[0]
        qubits = list(range(self.num_qubits))
        evo = pauli_evolution(self.num_qubits, [(label, qubits, angle)])
        circuit = QuantumCircuit._from_circuit_data(
            evo,
            legacy_qubits=True,
            name="def_pauli_rotation",
        )
        self.definition = circuit

    def pauli(self) -> Pauli:
        """Return the Pauli rotation axis.

        Note that this does not include any potential sign in the :class:`.Pauli` object
        used to construct this gate, since the sign is absorbed into the rotation angle,
        which is accessible via the ``params`` attribute.

        Returns:
            The Pauli rotation axis.
        """
        from qiskit.quantum_info import Pauli  # pylint: disable=cyclic-import

        return Pauli((self._pauli_z, self._pauli_x, 0))

    def to_matrix(self):
        pauli_matrix = self.pauli().to_matrix(sparse=False)
        angle = self.params[0]
        return sc.linalg.expm(-1j * angle / 2 * pauli_matrix)
