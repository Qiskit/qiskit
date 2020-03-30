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

"""Optimize chains of single-qubit u1, u2, u3 gates by combining them into a single gate."""

from itertools import groupby

import numpy as np

from qiskit.transpiler.exceptions import TranspilerError
from qiskit.extensions.standard.u1 import U1Gate
from qiskit.extensions.standard.u2 import U2Gate
from qiskit.extensions.standard.u3 import U3Gate
from qiskit.circuit.gate import Gate
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.quantum_info.operators import Quaternion

_CHOP_THRESHOLD = 1e-15

class Node:
    """
    Operator description in Optimize1qGates(TransformationPass)
    """
    def __init__(self, name, parameters):
        """
        # right.name = "gate name in (u1, u2, u3)"
        # right.parameters = (theta, phi, lambda)
        """
        self.name = name
        self.parameters = parameters


class Optimize1qGates(TransformationPass):
    """Optimize chains of single-qubit u1, u2, u3 gates by combining them into a single gate."""

    def define_left_node(self, current_node):
        """
        Create left_node from current_node
        """
        left_name = current_node.name
        if (current_node.condition is not None
                or len(current_node.qargs) != 1
                or left_name not in ["u1", "u2", "u3", "id"]):
            raise TranspilerError("internal error")
        if left_name == "u1":
            left_parameters = (0, 0, current_node.op.params[0])
        elif left_name == "u2":
            left_parameters = (np.pi / 2, current_node.op.params[0],
                               current_node.op.params[1])
        elif left_name == "u3":
            left_parameters = tuple(current_node.op.params)
        else:
            left_name = "u1"  # replace id with u1
            left_parameters = (0, 0, 0)
        # If there are any sympy objects coming from the gate convert
        # to numpy.
        left_parameters = tuple([float(x) for x in left_parameters])
        return Node(left_name, left_parameters)

    def combine_u1u1_into_right_node(self, left_node, right_node):
        """
        u1(lambda1) * u1(lambda2) = u1(lambda1 + lambda2)
        """
        right_node.parameters = (0, 0, right_node.parameters[2] +
                                 left_node.parameters[2])

    def combine_u1u2_into_right_node(self, left_node, right_node):
        """
        u1(lambda1) * u2(phi2, lambda2) = u2(phi2 + lambda1, lambda2)
        """
        right_node.parameters = (np.pi / 2, right_node.parameters[1] +
                                 left_node.parameters[2], right_node.parameters[2])

    def combine_u1u3_into_right_node(self, left_node, right_node):
        """
        u1(lambda1) * u3(theta2, phi2, lambda2) =
             u3(theta2, phi2 + lambda1, lambda2)
        """
        right_node.parameters = (right_node.parameters[0], right_node.parameters[1] +
                                 left_node.parameters[2], right_node.parameters[2])

    def combine_u2u1_into_right_node(self, left_node, right_node):
        """
        u2(phi1, lambda1) * u1(lambda2) = u2(phi1, lambda1 + lambda2)
        """
        right_node.name = "u2"
        right_node.parameters = (np.pi / 2, left_node.parameters[1],
                                 right_node.parameters[2] + left_node.parameters[2])

    def combine_u2u2_into_right_node(self, left_node, right_node):
        """
        Using Ry(pi/2).Rz(2*lambda).Ry(pi/2) =
           Rz(pi/2).Ry(pi-2*lambda).Rz(pi/2),
        u2(phi1, lambda1) * u2(phi2, lambda2) =
            u3(pi - lambda1 - phi2, phi1 + pi/2, lambda2 + pi/2)
        """
        right_node.name = "u3"
        right_node.parameters = (np.pi - left_node.parameters[2] -
                                 right_node.parameters[1], left_node.parameters[1] +
                                 np.pi / 2, right_node.parameters[2] +
                                 np.pi / 2)

    def combine_u3u1_into_right_node(self, left_node, right_node):
        """
        u3(theta1, phi1, lambda1) * u1(lambda2) =
            u3(theta1, phi1, lambda1 + lambda2)
        """
        right_node.name = "u3"
        right_node.parameters = (left_node.parameters[0], left_node.parameters[1],
                                 right_node.parameters[2] + left_node.parameters[2])

    def combine_u2oru3u3_into_right_node(self, left_node, right_node):
        """
        For composing u3's or u2's with u3's, use
        u2(phi, lambda) = u3(pi/2, phi, lambda)
        together with the qiskit.mapper.compose_u3 method.
        """
        right_node.name = "u3"
        # Evaluate the symbolic expressions for efficiency
        right_node.parameters = Optimize1qGates.compose_u3(left_node.parameters[0],
                                                           left_node.parameters[1],
                                                           left_node.parameters[2],
                                                           right_node.parameters[0],
                                                           right_node.parameters[1],
                                                           right_node.parameters[2])

    def combine_gates_into_right_node(self, left_node, right_node):
        """
        Combine left_node and right_node into right_node
        """
        name_tuple = (left_node.name, right_node.name)
        if name_tuple == ("u1", "u1"):
            self.combine_u1u1_into_right_node(left_node, right_node)

        elif name_tuple == ("u1", "u2"):
            self.combine_u1u2_into_right_node(left_node, right_node)

        elif name_tuple == ("u2", "u1"):
            self.combine_u2u1_into_right_node(left_node, right_node)

        elif name_tuple == ("u1", "u3"):
            self.combine_u1u3_into_right_node(left_node, right_node)

        elif name_tuple == ("u3", "u1"):
            self.combine_u3u1_into_right_node(left_node, right_node)

        elif name_tuple == ("u2", "u2"):
            self.combine_u2u2_into_right_node(left_node, right_node)

        elif name_tuple[1] == "nop":
            right_node.name = left_node.name
            right_node.parameters = left_node.parameters
        else:
            self.combine_u2oru3u3_into_right_node(left_node, right_node)

    def generate_gate(self, right_node):
        """
        Generate new gate from right_node
        """
        new_op = Gate(name="", num_qubits=1, params=[])
        if right_node.name == "u1":
            new_op = U1Gate(right_node.parameters[2])
        if right_node.name == "u2":
            new_op = U2Gate(right_node.parameters[1], right_node.parameters[2])
        if right_node.name == "u3":
            new_op = U3Gate(*right_node.parameters)
        return new_op

    def add_right_node_to_dag(self, dag, right_node, run):
        """
        Add right_node to dag
        """
        for k, operator in enumerate(right_node):
            dag.substitute_node(run[k], operator, inplace=True)

    def extract_simplified_gates(self, dag, run, num_gates):
        """
        Delete the other nodes in the run
        """
        for current_node in run[num_gates:]:
            dag.remove_op_node(current_node)
        # if right_node.name == "nop":
        #     dag.remove_op_node(run[0])

    def can_combine(self, left_node, right_node):
        """
            Check if left_node and right_node can be combined preserving global phase
        """
        if left_node.name == 'u1' or right_node.name == 'u1':
            return True
        if left_node.name == 'u2' and right_node.name == 'u2':
            temp = - 0.5 * np.e**(left_node.parameters[2]*1j) * \
                   np.e**(right_node.parameters[1]*1j)+0.5
            if np.abs(np.imag(temp) < _CHOP_THRESHOLD) and np.real(temp) >= 0:
                return True

        if left_node.name == 'u2' and right_node.name == 'u3':
            temp = - 0.707106781186547 * \
                   np.e**((left_node.parameters[2] + right_node.parameters[1])*1j) * \
                   np.sin(right_node.parameters[0] / 2) + \
                   0.707106781186548 * np.cos(right_node.parameters[0]/2)

            if np.abs(np.imag(temp) < _CHOP_THRESHOLD) and np.real(temp) >= 0:
                return True

        if left_node.name == 'u3' and right_node.name == 'u2':
            temp = - 0.707106781186547 * \
                   np.e**((left_node.parameters[2] + right_node.parameters[1])*1j) \
                   * np.sin(left_node.parameters[0] / 2) + \
                   0.707106781186548 * np.cos(left_node.parameters[0]/2)
            if np.abs(np.imag(temp) < _CHOP_THRESHOLD) and np.real(temp) >= 0:
                return True

        if left_node.name == 'u3' and right_node.name == 'u3':
            temp = -np.e**((left_node.parameters[2] + right_node.parameters[1])*1j) \
                   * np.sin(left_node.parameters[0] / 2) * np.sin(right_node.parameters[0] / 2) + \
                   np.cos(left_node.parameters[0]/2) * np.cos(right_node.parameters[0]/2)
            if np.abs(np.imag(temp) < _CHOP_THRESHOLD) and np.real(temp) >= 0:
                return True

        return False

    def run(self, dag):
        """Run the Optimize1qGates pass on `dag`.

                Args:
                    dag (DAGCircuit): the DAG to be optimized.

                Returns:
                    DAGCircuit: the optimized DAG.

                Raises:
                    TranspilerError: if YZY and ZYZ angles do not give same rotation matrix.
                """

        runs = dag.collect_runs(["u1", "u2", "u3"])
        runs = _split_runs_on_parameters(runs)

        for run in runs:
            num_gates = 0 # pylint: disable = attribute-defined-outside-init
            final_gates = []
            final_nodes = []
            right_node = Node("u1", (0, 0, 0))

            for current_node in run:
                left_node = self.define_left_node(current_node)
                if self.can_combine(left_node, right_node):
                    self.combine_gates_into_right_node(left_node, right_node)
                else:
                    new_op = self.generate_gate(right_node)
                    final_nodes.append(Node(right_node.name, right_node.parameters))
                    final_gates.append(new_op)
                    right_node.name = left_node.name
                    right_node.parameters = left_node.parameters
                    num_gates = num_gates + 1


                self.simplify_right_node(right_node)


            new_op = self.generate_gate(right_node)
            final_gates.append(new_op)
            num_gates = num_gates + 1
            self.add_right_node_to_dag(dag, final_gates, run)
            self.extract_simplified_gates(dag, run, num_gates)
            self.check(run, dag)


        return dag

    def check(self, run, dag):
        """
        Remove Null operators
        """
        for node_op in run:
            if node_op.name == '':
                dag.remove_op_node(node_op)


    def simplify_right_node(self, right_node):
        """
        1. Here down, when we simplify, we add f(theta) to lambda to
        correct the global phase when f(theta) is 2*pi. This isn't
        necessary but the other steps preserve the global phase, so
        we continue in that manner.
        2. The final step will remove Z rotations by 2*pi.
        3. Note that is_zero is true only if the expression is exactly
        zero. If the input expressions have already been evaluated
        then these final simplifications will not occur.
        TODO After we refactor, we should have separate passes for
        exact and approximate rewriting.
        Y rotation is 0 mod 2*pi, so the gate is a u1
        """
        if np.mod(right_node.parameters[0], (2 * np.pi)) == 0 \
                and right_node.name != "u1":
            right_node.name = "u1"
            right_node.parameters = (0, 0, right_node.parameters[1] +
                                     right_node.parameters[2] +
                                     right_node.parameters[0])
        # Y rotation is pi/2 or -pi/2 mod 2*pi, so the gate is a u2
        if right_node.name == "u3":
            # theta = pi/2 + 2*k*pi
            if np.mod((right_node.parameters[0] - np.pi / 2), (2 * np.pi)) == 0:
                right_node.name = "u2"
                right_node.parameters = (np.pi / 2, right_node.parameters[1],
                                         right_node.parameters[2] +
                                         (right_node.parameters[0] - np.pi / 2))
            # theta = -pi/2 + 2*k*pi
            if np.mod((right_node.parameters[0] + np.pi / 2), (2 * np.pi)) == 0:
                right_node.name = "u2"
                right_node.parameters = (np.pi / 2, right_node.parameters[1] +
                                         np.pi, right_node.parameters[2] -
                                         np.pi + (right_node.parameters[0] +
                                                  np.pi / 2))
        # u1 and lambda is 0 mod 2*pi so gate is nop (up to a global phase)
        if right_node.name == "u1" and np.mod(right_node.parameters[2], (2 * np.pi)) == 0:
            right_node.name = "nop"

    @staticmethod
    def compose_u3(theta1, phi1, lambda1, theta2, phi2, lambda2):
        """Return a triple theta, phi, lambda for the product.

        u3(theta, phi, lambda)
           = u3(theta1, phi1, lambda1).u3(theta2, phi2, lambda2)
           = Rz(phi1).Ry(theta1).Rz(lambda1+phi2).Ry(theta2).Rz(lambda2)
           = Rz(phi1).Rz(phi').Ry(theta').Rz(lambda').Rz(lambda2)
           = u3(theta', phi1 + phi', lambda2 + lambda')

        Return theta, phi, lambda.
        """
        # Careful with the factor of two in yzy_to_zyz
        thetap, phip, lambdap = Optimize1qGates.yzy_to_zyz((lambda1 + phi2), theta1, theta2)
        (theta, phi, lamb) = (thetap, phi1 + phip, lambda2 + lambdap)

        return (theta, phi, lamb)

    @staticmethod
    def yzy_to_zyz(xi, theta1, theta2, eps=1e-9):  # pylint: disable=invalid-name
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
        out_angles = tuple(0 if np.abs(angle) < _CHOP_THRESHOLD else angle
                           for angle in out_angles)
        return out_angles


def _split_runs_on_parameters(runs):
    """Finds runs containing parameterized gates and splits them into sequential
    runs excluding the parameterized gates.
    """

    out = []
    for run in runs:
        groups = groupby(run, lambda x: x.op.is_parameterized())

        for group_is_parameterized, gates in groups:
            if not group_is_parameterized:
                out.append(list(gates))

    return out
