# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Simulator command to snapshot internal simulator representation.
"""

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.instruction import Instruction
from qiskit.extensions.exceptions import QiskitError, ExtensionError


class Snapshot(Instruction):
    """Simulator snapshot instruction."""

    def __init__(self,
                 label,
                 snapshot_type='statevector',
                 num_qubits=0,
                 num_clbits=0,
                 params=None):
        """Create new snapshot instruction.

        Args:
            label (str): the snapshot label for result data.
            snapshot_type (str): the type of the snapshot.
            num_qubits (int): the number of qubits for the snapshot type [Default: 0].
            num_clbits (int): the number of classical bits for the snapshot type [Default: 0].
            params (list or None): the parameters for snapshot_type [Default: None].

        Raises:
            ExtensionError: if snapshot label is invalid.
        """
        if not isinstance(label, str):
            raise ExtensionError('Snapshot label must be a string.')
        self._label = label
        self._snapshot_type = snapshot_type
        if params is None:
            params = []
        super().__init__('snapshot', num_qubits, num_clbits, params)

    def assemble(self):
        """Assemble a QasmQobjInstruction"""
        instruction = super().assemble()
        instruction.label = self._label
        instruction.snapshot_type = self._snapshot_type
        return instruction

    def inverse(self):
        """Special case. Return self."""
        return Snapshot(self.num_qubits, self.num_clbits, self.params[0],
                        self.params[1])

    @property
    def snapshot_type(self):
        """Return snapshot type"""
        return self._snapshot_type

    @property
    def label(self):
        """Return snapshot label"""
        return self._label

    @label.setter
    def label(self, name):
        """Set snapshot label to name

        Args:
            name (str or None): label to assign unitary

        Raises:
            TypeError: name is not string or None.
        """
        if isinstance(name, str):
            self._label = name
        else:
            raise TypeError('label expects a string')

    def c_if(self, classical, val):
        raise QiskitError('Snapshots are simulator directives and cannot be conditional.')


def snapshot(self,
             label,
             snapshot_type='statevector',
             qubits=None,
             params=None):
    """Take a statevector snapshot of the internal simulator representation.
    Works on all qubits, and prevents reordering (like barrier).

    For other types of snapshots use the Snapshot extension directly.

    Args:
        label (str): a snapshot label to report the result.
        snapshot_type (str): the type of the snapshot.
        qubits (list or None): the qubits to apply snapshot to [Default: None].
        params (list or None): the parameters for snapshot_type [Default: None].

    Returns:
        QuantumCircuit: with attached command

    Raises:
        ExtensionError: malformed command
    """
    # If no qubits are specified we add all qubits so it acts as a barrier
    # This is needed for full register snapshots like statevector
    if isinstance(qubits, QuantumRegister):
        qubits = qubits[:]
    if not qubits:
        tuples = []
        if isinstance(self, QuantumCircuit):
            for register in self.qregs:
                tuples.append(register)
        if not tuples:
            raise ExtensionError('no qubits for snapshot')
        qubits = []
        for tuple_element in tuples:
            if isinstance(tuple_element, QuantumRegister):
                for j in range(tuple_element.size):
                    qubits.append(tuple_element[j])
            else:
                qubits.append(tuple_element)
    return self.append(
        Snapshot(
            label,
            snapshot_type=snapshot_type,
            num_qubits=len(qubits),
            params=params), qubits)


# Add to QuantumCircuit class
QuantumCircuit.snapshot = snapshot
