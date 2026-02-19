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

from qiskit.circuit import QuantumCircuit, CircuitError, Gate, ParameterExpression
from qiskit._accelerate.circuit_library import pauli_evolution

if typing.TYPE_CHECKING:
    from qiskit.quantum_info import Pauli
    from qiskit.circuit.quantumcircuit import ParameterValueType


class PauliRotationGate(Gate):
    r"""A generic Pauli rotation.

    This implements the unitary operation

    .. math::

        e^{i \theta / 2 P}

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
            pauli: A tensor product of Pauli operators defining the measurement,
                for example ``Pauli("XY")`` or ``Pauli("-XYIZ")``.
                The identity Pauli operator is not permitted.
                The Pauli may include a phase of :math:`-1`, but not :math:`i` or :math:`-i`.
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
            label = _get_default_label(pauli, params[0])

        super().__init__(
            name="pauli_rotation",
            num_qubits=num_qubits,
            params=params,
            label=label,
        )

    @classmethod
    def _from_pauli_data(cls, z, x, angle, label):
        """
        Instantiates a PauliRotationGate from raw pauli data, the angle, and the label.
        This function is used internally from within the rust code and from QPY
        serialization.
        """
        from qiskit.quantum_info import Pauli  # pylint: disable=cyclic-import

        return cls(Pauli((z, x, 0)), angle, label)

    def inverse(self, annotated=False):
        """Prevents from calling ``inverse`` on a PauliProductMeasurement instruction."""
        raise CircuitError("PauliProductMeasurement is not invertible.")

    def __eq__(self, other):
        if not isinstance(other, PauliRotationGate):
            return False

        return (
            np.all(self._pauli_z == other._pauli_z)
            and np.all(self._pauli_x == other._pauli_x)
            and (self.params[0] == other.params[0])
        )

    def _define(self):
        label = self.pauli.to_label()
        evo = pauli_evolution(self.num_qubits, [label])
        circuit = QuantumCircuit._from_circuit_data(
            evo,
            legacy_qubits=True,
            name="def_ppm",
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
        return Pauli((self._pauli_z, self._pauli_x, 0))


def _get_default_label(pauli, angle) -> str:
    """Creates the default label for PauliProductMeasurement instruction,
    used for visualization.
    """
    # If the angle is a float, truncate the number of digits we print
    # TODO use pi check
    if isinstance(angle, ParameterExpression):
        angle = str(angle)
    else:
        angle = f"{angle:.3f}"

    return f"P({pauli.to_label()}, {angle})"


def _is_identity_pauli(pauli: Pauli):
    """Return whether a Pauli has an all-'I' label (up to a phase)."""
    return not np.any(pauli.z) and not np.any(pauli.x)
