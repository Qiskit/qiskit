# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The standard gates moved to qiskit/circuit/library."""
    def control(self, num_ctrl_qubits=1, label=None, ctrl_state=None):
        """Controlled version of this gate.

        Args:
            num_ctrl_qubits (int): number of control qubits.
            label (str or None): An optional label for the gate [Default: None]
            ctrl_state (int or str or None): control state expressed as integer,
                string (e.g. '110'), or None. If None, use all 1s.

        Returns:
            Gate: Generic gate with all identities.
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit import QuantumRegister
        if num_ctrl_qubits > 2:
            ctrl_substr = 'c{0:d}'.format(num_ctrl_qubits)
        else:
            ctrl_substr = ('{0}' * num_ctrl_qubits).format('c')
        new_name = '{0}{1}'.format(ctrl_substr, self.name)
        num_qubits = 1 + num_ctrl_qubits
        idgate = Gate(new_name, num_qubits, [], label=label)
        qr = QuantumRegister(num_qubits, 'q')
        idgate._definition = [(IGate(), [qr[i]], []) for i in range(num_qubits)]
        return idgate


from qiskit.circuit.library.standard_gates.i import IGate, IdGate

__all__ = ['IGate', 'IdGate']
