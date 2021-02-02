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

from typing import List
import numpy as np

from qiskit.circuit import Parameter, QuantumCircuit


def rzx_zz1(theta: float = None):
    """ZZ template with rz gate."""
    if theta is None:
        theta = Parameter('ϴ')

    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.rz(theta, 1)
    qc.sx(1)
    qc.rz(np.pi, 1)
    qc.sx(1)
    qc.rz(3 * np.pi, 1)
    qc.cx(0, 1)
    qc.rz(-1*theta, 1)

    # Hadamard
    qc.rz(np.pi / 2, 1)
    qc.rx(np.pi / 2, 1)
    qc.rz(np.pi / 2, 1)

    qc.rx(theta, 1)
    qc.rzx(-1*theta, 0, 1)
    # Hadamard
    qc.rz(np.pi / 2, 1)
    qc.rx(np.pi / 2, 1)
    qc.rz(np.pi / 2, 1)

    return qc


def rzx_zz2(theta: float = None):
    """ZZ template is p gate."""
    if theta is None:
        theta = Parameter('ϴ')

    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.p(theta, 1)
    qc.cx(0, 1)
    qc.p(-1*theta, 1)
    # Hadamard
    qc.rz(np.pi / 2, 1)
    qc.rx(np.pi / 2, 1)
    qc.rz(np.pi / 2, 1)

    qc.rx(theta, 1)
    qc.rzx(-1*theta, 0, 1)
    # Hadamard
    qc.rz(np.pi / 2, 1)
    qc.rx(np.pi / 2, 1)
    qc.rz(np.pi / 2, 1)

    return qc


def rzx_zz3(theta: float = None):
    """ZZ template is p gate."""
    if theta is None:
        theta = Parameter('ϴ')

    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.rz(theta, 1)
    qc.cx(0, 1)
    qc.rz(-1*theta, 1)
    # Hadamard
    qc.rz(np.pi / 2, 1)
    qc.rx(np.pi / 2, 1)
    qc.rz(np.pi / 2, 1)

    qc.rx(theta, 1)
    qc.rzx(-1*theta, 0, 1)
    # Hadamard
    qc.rz(np.pi / 2, 1)
    qc.rx(np.pi / 2, 1)
    qc.rz(np.pi / 2, 1)

    return qc


def rzx_xz(theta: float = None):
    """ZY template."""
    if theta is None:
        theta = Parameter('ϴ')

    qc = QuantumCircuit(2)
    qc.cx(1, 0)
    qc.rx(theta, 1)
    qc.cx(1, 0)

    qc.rz(np.pi / 2, 0)
    qc.rx(np.pi / 2, 0)
    qc.rz(np.pi / 2, 0)
    qc.rzx(-1*theta, 0, 1)
    qc.rz(np.pi / 2, 0)
    qc.rx(np.pi / 2, 0)
    qc.rz(np.pi / 2, 0)
    return qc


def rzx_yz(theta: float = None):
    """ZY template."""
    if theta is None:
        theta = Parameter('ϴ')

    circ = QuantumCircuit(2)
    circ.cx(0, 1)
    circ.ry(-1*theta, 0)
    circ.cx(0, 1)
    circ.rx(np.pi / 2, 0)
    circ.rzx(theta, 0, 1)
    circ.rx(-np.pi / 2, 0)

    return circ


def rzx_cy(theta: float = None):
    """ZY template."""
    if theta is None:
        theta = Parameter('ϴ')

    circ = QuantumCircuit(2)
    circ.cx(0, 1)
    circ.ry(theta, 1)
    circ.cx(0, 1)
    circ.ry(-1*theta, 1)
    circ.rz(-np.pi / 2, 1)
    circ.rx(theta, 1)
    circ.rzx(-1*theta, 0, 1)
    circ.rz(np.pi / 2, 1)

    return circ


def rzx_templates(template_list: List[str] = None):
    """
    Convenience function to get the cost_dict and
    templates for template matching.
    """

    if template_list is None:
        template_list = ['zz1', 'zz2', 'zz3', 'yz', 'xz', 'cy']

    templates = []
    if 'zz1' in template_list:
        templates.append(rzx_zz1())
    if 'zz2' in template_list:
        templates.append(rzx_zz2())
    if 'zz3' in template_list:
        templates.append(rzx_zz3())
    if 'yz' in template_list:
        templates.append(rzx_yz())
    if 'xz' in template_list:
        templates.append(rzx_xz())
    if 'cy' in template_list:
        templates.append(rzx_cy())

    cost_dict = {'rzx': 0, 'cx': 6, 'rz': 0, 'sx': 1, 'p': 0, 'h': 1, 'rx': 1, 'ry': 1}

    rzx_dict = {'template_list': templates, 'user_cost_dict': cost_dict}

    return rzx_dict
