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

from qiskit.circuit.library.standard_gates.x import (
    XGate, CXGate, CCXGate, C3XGate, C4XGate, RCCXGate, RC3XGate, CnotGate, ToffoliGate,
    MCXGate, MCXGrayCode, MCXVChain, MCXRecursive
)
    def control(self, num_ctrl_qubits=1, label=None, ctrl_state=None):
        """Controlled version of this gate. This is needed for distinguishing CnotGate
        from an opaque gate.

        Args:
            num_ctrl_qubits (int): number of control qubits.
            label (str or None): An optional label for the gate [Default: None]
            ctrl_state (int or str or None): control state expressed as integer,
                string (e.g. '110'), or None. If None, use all 1s.

        Returns:
            ControlledGate: controlled version of this gate.
        """
        return super().control(num_ctrl_qubits=num_ctrl_qubits, label=label,
                               ctrl_state=ctrl_state)


__all__ = ['XGate', 'CXGate', 'CCXGate', 'C3XGate', 'C4XGate', 'RCCXGate', 'RC3XGate',
           'CnotGate', 'ToffoliGate', 'MCXGate', 'MCXGrayCode', 'MCXVChain', 'MCXRecursive']
