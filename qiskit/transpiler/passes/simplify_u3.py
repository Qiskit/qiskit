# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A strength reduction pass to simplify single qubit U3 gates, if possible.
"""

import numpy as np

from qiskit.extensions.standard.u1 import U1Gate
from qiskit.extensions.standard.u2 import U2Gate
from qiskit.extensions.standard.u3 import U3Gate
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.add_control import add_control
from qiskit.transpiler.basepasses import TransformationPass


DEFAULT_ATOL = 1e-12


class SimplifyU3(TransformationPass):
    """A strength reduction pass to simplify single qubit U3 gates, if possible.

    The cost metric is the number of X90 pulses required to implement the gate.
    Can convert U3 -> U2 OR U1 OR None. 
    Also makes all Euler angles modulo 2*pi.

    Additional Information
    ----------------------
    U3(θ,φ,λ) is a generic single-qubit gate (generic Bloch sphere rotation).
    It can be realized with TWO pre-calibrated X90 pulses.
    Example: X gate.

    U2(φ,λ) is a rotation around the Z+X axis (with co-efficient). It can
    be implemented using ONE pre-calibrated X90 pulse.
    Example: H gate.

    U1(λ) is a rotation about the Z axis. It requires ZERO pulses (i.e.
    done virtually in software).
    Example: T gate.
    """

    def run(self, dag):
        """Run the SimplifyU3 pass on `dag`.

        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: the optimized DAG.
        """
        for node in dag.gate_nodes():
            op = node.op
            num_ctrl_qubits = None

            if isinstance(node.op, ControlledGate):
                num_ctrl_qubits = op.num_ctrl_qubits
                op = node.op.base_gate

            if isinstance(op, U3Gate):
                theta, phi, lam = op.params

                new_op = U3Gate(_mod2pi(theta), _mod2pi(phi), _mod2pi(lam))

                if np.allclose([theta, phi, lam], [0., 0., 0.], atol=DEFAULT_ATOL):
                    new_op = None

                elif np.allclose([theta], [0.], atol=DEFAULT_ATOL):
                    new_op = U1Gate(_mod2pi(phi+lam))

                elif np.isclose(theta, np.pi/2, atol=DEFAULT_ATOL):
                    new_op = U2Gate(_mod2pi(phi), _mod2pi(lam))

                elif np.isclose(theta, 3*np.pi/2, atol=DEFAULT_ATOL):
                    new_op = U2Gate(_mod2pi(phi+np.pi), _mod2pi(lam+np.pi))

                if new_op is None:
                    dag.remove_op_node(node)
                else:
                    if num_ctrl_qubits is not None:
                       new_op = add_control(new_op, num_ctrl_qubits)
                    dag.substitute_node(node, new_op)

        return dag


def _mod2pi(angle):
    return np.mod(angle, 2*np.pi)
