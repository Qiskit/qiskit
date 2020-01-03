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

"""Collapse chains of single qubit gates into a single U3 gate.
"""

from itertools import groupby

import numpy as np

from qiskit.transpiler.exceptions import TranspilerError
from qiskit.extensions.standard.u1 import U1Gate
from qiskit.extensions.standard.u2 import U2Gate
from qiskit.extensions.standard.u3 import U3Gate
from qiskit.circuit.gate import Gate
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.quantum_info.operators import Quaternion, Operator
from qiskit.quantum_info.synthesis.one_qubit_decompose import OneQubitEulerDecomposer

DEFAULT_ATOL = 1e-15


class Collapse1qChains(TransformationPass):
    """Collapse chains of single-qubit gates into a single U3 gate each.

    A single gate (chain of one) will also be converted.

    This pass can make some gates temporarily less optimized, for example
    by converting a U1 gate into a U3. A follow-up invocation of SimplifyU3
    can correct for this.
    """

    def run(self, dag):
        """Run the Collapse1qChains pass on `dag`.

        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: a DAG with no single-qubit gate chains and only U3s
                as single-qubit gates.

        Raises:
            TranspilerError: in case of numerical errors in combining U3 gates.
        """
        chains = []

        # collect chains of uninterrupted single-qubit gates
        topo_ops = list(dag.topological_op_nodes())
        nodes_seen = dict(zip(topo_ops, [False] * len(topo_ops)))
        for node in topo_ops:
            if isinstance(node.op, Gate) and \
                    len(node.qargs) == 1 and \
                    node.condition is None \
                    and not nodes_seen[node]:
                chain = [node]
                nodes_seen[node] = True
                successor = list(dag.successors(node))[0]  # has only one successor
                while successor.type == "op" and \
                        isinstance(successor.op, Gate) and \
                        len(successor.qargs) == 1 and \
                        successor.condition is None:
                    chain.append(successor)
                    nodes_seen[successor] = True
                    successor = list(dag.successors(successor))[0]
                chains.append(chain)

        # cannot collapse parameterized gates yet
        chains = _split_chains_on_parameters(chains)

        # collapse chains into a single U3
        decomposer = OneQubitEulerDecomposer(basis='U3')
        for chain in chains:
            left_parameters = (0, 0, 0)  # theta, phi, lambda
            for gate in reversed(chain):
                right_parameters = decomposer(Operator(gate.op)).data[0][0].params
                left_parameters = _compose_u3(left_parameters[0],
                                               left_parameters[1],
                                               left_parameters[2],
                                               right_parameters[0],
                                               right_parameters[1],
                                               right_parameters[2])

            new_op = U3Gate(*left_parameters)
            dag.substitute_node(chain[0], new_op, inplace=True)
            for node in chain[1:]:
                dag.remove_op_node(node)

        return dag


def _compose_u3(theta1, phi1, lambda1, theta2, phi2, lambda2):
    """Return a triple theta, phi, lambda for the product.

    u3(theta, phi, lambda)
       = u3(theta1, phi1, lambda1).u3(theta2, phi2, lambda2)
       = Rz(phi1).Ry(theta1).Rz(lambda1+phi2).Ry(theta2).Rz(lambda2)
       = Rz(phi1).Rz(phi').Ry(theta').Rz(lambda').Rz(lambda2)
       = u3(theta', phi1 + phi', lambda2 + lambda')

    Return theta, phi, lambda.
    """
    # Careful with the factor of two in yzy_to_zyz
    thetap, phip, lambdap = _yzy_to_zyz((lambda1 + phi2), theta1, theta2)
    (theta, phi, lamb) = (thetap, phi1 + phip, lambda2 + lambdap)
    return (theta, phi, lamb)


def _yzy_to_zyz(xi, theta1, theta2, eps=1e-9):  # pylint: disable=invalid-name
    """Express a Y.Z.Y single qubit gate as a Z.Y.Z gate.

    Solve the equation

    .. math::

    Ry(theta1).Rz(xi).Ry(theta2) = Rz(phi).Ry(theta).Rz(lambda)

    for theta, phi, and lambda.

    Return a solution theta, phi, and lambda.
    """
    quaternion_yzy = Quaternion.from_euler([theta1, xi, theta2], 'yzy')
    euler = quaternion_yzy.to_zyz()
    quaternion_zyz = Quaternion.from_euler(euler, 'zyz')
    # output order different than rotation order
    out_angles = (euler[1], euler[0], euler[2])
    abs_inner = abs(quaternion_zyz.data.dot(quaternion_yzy.data))
    if not np.allclose(abs_inner, 1, eps):
        raise TranspilerError('YZY and ZYZ angles do not give same rotation matrix.')
    out_angles = tuple(0 if np.abs(angle) < DEFAULT_ATOL else angle
                       for angle in out_angles)
    return out_angles


def _split_chains_on_parameters(chains):
    """Finds chains containing parameterized gates, and splits them into
    sequential chains excluding the parameterized gates.
    """
    out = []
    for chain in chains:
        groups = groupby(chain, lambda x: x.op.is_parameterized())

        for group_is_parameterized, gates in groups:
            if not group_is_parameterized:
                out.append(list(gates))
    return out
