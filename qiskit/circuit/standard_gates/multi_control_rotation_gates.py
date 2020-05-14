# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Multiple-Controlled U3 gate. Not using ancillary qubits.
"""

import logging
from qiskit.circuit.standard_gates.u3 import _generate_gray_code

logger = logging.getLogger(__name__)


def apply_cu3(circuit, theta, phi, lam, control, target, use_basis_gates=True):
    """Apply CU3."""
    if use_basis_gates:
        circuit.u1((lam + phi) / 2, control)
        circuit.u1((lam - phi) / 2, target)
        circuit.cx(control, target)
        circuit.u3(-theta / 2, 0, -(phi + lam) / 2, target)
        circuit.cx(control, target)
        circuit.u3(theta / 2, phi, 0, target)
    else:
        circuit.cu3(theta, phi, lam, control, target)


def apply_mcu3_graycode(circuit, theta, phi, lam, ctls, tgt, use_basis_gates):
    """Apply multi-controlled u3 gate from ctls to tgt using graycode
    pattern with single-step angles theta, phi, lam."""

    n = len(ctls)

    gray_code = _generate_gray_code(n)
    last_pattern = None

    for pattern in gray_code:
        if '1' not in pattern:
            continue
        if last_pattern is None:
            last_pattern = pattern
        # find left most set bit
        lm_pos = list(pattern).index('1')

        # find changed bit
        comp = [i != j for i, j in zip(pattern, last_pattern)]
        if True in comp:
            pos = comp.index(True)
        else:
            pos = None
        if pos is not None:
            if pos != lm_pos:
                circuit.cx(ctls[pos], ctls[lm_pos])
            else:
                indices = [i for i, x in enumerate(pattern) if x == '1']
                for idx in indices[1:]:
                    circuit.cx(ctls[idx], ctls[lm_pos])
        # check parity and undo rotation
        if pattern.count('1') % 2 == 0:
            # inverse CU3: u3(theta, phi, lamb)^dagger = u3(-theta, -lam, -phi)
            apply_cu3(circuit, -theta, -lam, -phi, ctls[lm_pos], tgt,
                      use_basis_gates=use_basis_gates)
        else:
            apply_cu3(circuit, theta, phi, lam, ctls[lm_pos], tgt,
                      use_basis_gates=use_basis_gates)
        last_pattern = pattern
