# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
A library of ZX based two qubit gates.
"""
import numpy as np
from typing import List

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, Gate
from qiskit.extensions import UnitaryGate


class ZX(Gate):
    """CZ gates used for the test."""

    def __init__(self, params):
        super().__init__('ZX', 2, params)
        self.num_ctrl_qubits = 1
        self.ctrl_state = 1

    def inverse(self):
        theta = -self.params[0]
        inverse = UnitaryGate(
            [[np.cos(-0.5 * theta), 1.0j * np.sin(-0.5 * theta), 0, 0],
             [1.0j * np.sin(-0.5 * theta), np.cos(-0.5 * theta), 0, 0],
             [0, 0, np.cos(0.5 * theta), 1.0j * np.sin(0.5 * theta)],
             [0, 0, 1.0j * np.sin(0.5 * theta), np.cos(0.5 * theta)]])
        inverse.name = 'zx(%.3f)' % theta
        return inverse


def zx_zz1():
    """ZZ template with rz gate."""
    theta = Parameter('ϴ')

    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.rz(theta, 1)
    qc.sx(1)
    qc.rz(np.pi, 1)
    qc.sx(1)
    qc.rz(3 * np.pi, 1)
    qc.cx(0, 1)
    qc.p(-theta, 1)

    # Hadamard
    qc.rz(np.pi / 2, 1)
    qc.rx(np.pi / 2, 1)
    qc.rz(np.pi / 2, 1)

    qc.rx(theta, 1)
    qc.append(ZX([-theta]), [1, 0])
    # Hadamard
    qc.rz(np.pi / 2, 1)
    qc.rx(np.pi / 2, 1)
    qc.rz(np.pi / 2, 1)

    return qc


def zx_zz2():
    """ZZ template is p gate."""
    theta = Parameter('ϴ')

    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.p(theta, 1)
    qc.cx(0, 1)
    qc.p(-theta, 1)
    # Hadamard
    qc.rz(np.pi / 2, 1)
    qc.rx(np.pi / 2, 1)
    qc.rz(np.pi / 2, 1)

    qc.rx(theta, 1)
    qc.append(ZX([-theta]), [1, 0])
    # Hadamard
    qc.rz(np.pi / 2, 1)
    qc.rx(np.pi / 2, 1)
    qc.rz(np.pi / 2, 1)

    return qc


def zx_zy():
    """ZY template."""
    theta = Parameter('ϴ')

    circ = QuantumCircuit(2)
    circ.cx(0, 1)
    circ.ry(-theta, 0)
    circ.cx(0, 1)
    circ.rx(np.pi / 2, 0)
    circ.append(ZX([theta]), [1, 0])
    circ.rx(-np.pi / 2, 0)

    return circ


def zx_templates(template_list: List[str] = None):
    """
    Convenience function to get the cost_dict and
    templates for template matching.
    """

    if template_list is None:
        template_list = ['zz1', 'zz2', 'zy']

    templates = []
    if 'zz1' in template_list:
        templates.append(zx_zz1())
    if 'zz2' in template_list:
        templates.append(zx_zz2())
    if 'zy' in template_list:
        templates.append(zx_zy())

    cost_dict = {'ZX': 0, 'cx': 6, 'rz': 1, 'sx': 2, 'p': 0, 'h': 1, 'rx': 1, 'ry': 1}

    zx_dict = {'template_list': templates, 'user_cost_dict': cost_dict}

    return zx_dict
