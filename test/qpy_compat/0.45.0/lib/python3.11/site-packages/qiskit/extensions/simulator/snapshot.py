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

from qiskit.circuit.instruction import Instruction
from qiskit.extensions.exceptions import QiskitError, ExtensionError

from qiskit.utils.deprecation import deprecate_func


class Snapshot(Instruction):
    """Simulator snapshot instruction."""

    _directive = True

    @deprecate_func(
        since="0.45.0",
        additional_msg="The Snapshot instruction has been superseded by Qiskit Aer's save "
        "instructions, see "
        "https://qiskit.org/ecosystem/aer/apidocs/aer_library.html#saving-simulator-data.",
    )
    def __init__(self, label, snapshot_type="statevector", num_qubits=0, num_clbits=0, params=None):
        """Create new snapshot instruction.

        Args:
            label (str): the snapshot label for result data.
            snapshot_type (str): the type of the snapshot.
            num_qubits (int): the number of qubits for the snapshot type [Default: 0].
            num_clbits (int): the number of classical bits for the snapshot type
                              [Default: 0].
            params (list or None): the parameters for snapshot_type [Default: None].

        Raises:
            ExtensionError: if snapshot label is invalid.
        """
        if not isinstance(label, str):
            raise ExtensionError("Snapshot label must be a string.")
        self._snapshot_type = snapshot_type
        if params is None:
            params = []
        super().__init__("snapshot", num_qubits, num_clbits, params, label=label)

    def assemble(self):
        """Assemble a QasmQobjInstruction"""
        instruction = super().assemble()
        instruction.snapshot_type = self._snapshot_type
        return instruction

    def inverse(self):
        """Special case. Return self."""
        return Snapshot(self.num_qubits, self.num_clbits, self.params[0], self.params[1])

    @property
    def snapshot_type(self):
        """Return snapshot type"""
        return self._snapshot_type

    def c_if(self, classical, val):
        raise QiskitError("Snapshots are simulator directives and cannot be conditional.")
