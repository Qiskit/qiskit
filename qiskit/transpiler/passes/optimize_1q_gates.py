# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""
Transpiler pass to optimize chains of single-qubit u1, u2, u3 gates by combining them into
a single gate.
"""

import networkx as nx
import numpy as np
import sympy
from sympy import Number as N

from qiskit.mapper import MapperError
from qiskit.extensions.standard.u1 import U1Gate
from qiskit.extensions.standard.u2 import U2Gate
from qiskit.extensions.standard.u3 import U3Gate
from qiskit.circuit.instruction import Instruction
from qiskit.transpiler import TransformationPass
from qiskit.quantum_info.operators.quaternion import quaternion_from_euler
from qiskit.transpiler.passes.mapping.unroller import Unroller


_CHOP_THRESHOLD = 1e-15


class Optimize1qGates(TransformationPass):
    """Simplify runs of single qubit gates in the ["u1", "u2", "u3", "cx", "id"] basis."""

    def __init__(self):
        super().__init__()
        self.requires.append(Unroller(["u1", "u2", "u3", "cx", "id"]))

    def run(self, dag):
        """Return a new circuit that has been optimized."""
        runs = dag.collect_runs(["u1", "u2", "u3", "id"])
        for run in runs:
            run_qarg = dag.multi_graph.node[run[0]]["qargs"][0]
            right_name = "u1"
            right_parameters = (N(0), N(0), N(0))  # (theta, phi, lambda)
            for current_node in run:
                node = dag.multi_graph.node[current_node]
                left_name = node["name"]
                if (node["condition"] is not None
                        or len(node["qargs"]) != 1
                        or node["qargs"][0] != run_qarg
                        or left_name not in ["u1", "u2", "u3", "id"]):
                    raise MapperError("internal error")
                if left_name == "u1":
                    left_parameters = (N(0), N(0), node["op"].param[0])
                elif left_name == "u2":
                    left_parameters = (sympy.pi / 2, node["op"].param[0], node["op"].param[1])
                elif left_name == "u3":
                    left_parameters = tuple(node["op"].param)
                else:
                    left_name = "u1"  # replace id with u1
                    left_parameters = (N(0), N(0), N(0))
                # Compose gates
                name_tuple = (left_name, right_name)
                if name_tuple == ("u1", "u1"):
                    # u1(lambda1) * u1(lambda2) = u1(lambda1 + lambda2)
                    right_parameters = (N(0), N(0), right_parameters[2] +
                                        left_parameters[2])
                elif name_tuple == ("u1", "u2"):
                    # u1(lambda1) * u2(phi2, lambda2) = u2(phi2 + lambda1, lambda2)
                    right_parameters = (sympy.pi / 2, right_parameters[1] +
                                        left_parameters[2], right_parameters[2])
                elif name_tuple == ("u2", "u1"):
                    # u2(phi1, lambda1) * u1(lambda2) = u2(phi1, lambda1 + lambda2)
                    right_name = "u2"
                    right_parameters = (sympy.pi / 2, left_parameters[1],
                                        right_parameters[2] + left_parameters[2])
                elif name_tuple == ("u1", "u3"):
                    # u1(lambda1) * u3(theta2, phi2, lambda2) =
                    #     u3(theta2, phi2 + lambda1, lambda2)
                    right_parameters = (right_parameters[0], right_parameters[1] +
                                        left_parameters[2], right_parameters[2])
                elif name_tuple == ("u3", "u1"):
                    # u3(theta1, phi1, lambda1) * u1(lambda2) =
                    #     u3(theta1, phi1, lambda1 + lambda2)
                    right_name = "u3"
                    right_parameters = (left_parameters[0], left_parameters[1],
                                        right_parameters[2] + left_parameters[2])
                elif name_tuple == ("u2", "u2"):
                    # Using Ry(pi/2).Rz(2*lambda).Ry(pi/2) =
                    #    Rz(pi/2).Ry(pi-2*lambda).Rz(pi/2),
                    # u2(phi1, lambda1) * u2(phi2, lambda2) =
                    #    u3(pi - lambda1 - phi2, phi1 + pi/2, lambda2 + pi/2)
                    right_name = "u3"
                    right_parameters = (sympy.pi - left_parameters[2] -
                                        right_parameters[1], left_parameters[1] +
                                        sympy.pi / 2, right_parameters[2] +
                                        sympy.pi / 2)
                elif name_tuple[1] == "nop":
                    right_name = left_name
                    right_parameters = left_parameters
                else:
                    # For composing u3's or u2's with u3's, use
                    # u2(phi, lambda) = u3(pi/2, phi, lambda)
                    # together with the qiskit.mapper.compose_u3 method.
                    right_name = "u3"
                    # Evaluate the symbolic expressions for efficiency
                    left_parameters = tuple(map(lambda x: x.evalf(), list(left_parameters)))
                    right_parameters = tuple(map(lambda x: x.evalf(), list(right_parameters)))
                    right_parameters = Optimize1qGates.compose_u3(left_parameters[0],
                                                                  left_parameters[1],
                                                                  left_parameters[2],
                                                                  right_parameters[0],
                                                                  right_parameters[1],
                                                                  right_parameters[2])
                    # Why evalf()? This program:
                    #   OPENQASM 2.0;
                    #   include "qelib1.inc";
                    #   qreg q[2];
                    #   creg c[2];
                    #   u3(0.518016983430947*pi,1.37051598592907*pi,1.36816383603222*pi) q[0];
                    #   u3(1.69867232277986*pi,0.371448347747471*pi,0.461117217930936*pi) q[0];
                    #   u3(0.294319836336836*pi,0.450325871124225*pi,1.46804720442555*pi) q[0];
                    #   measure q -> c;
                    # took >630 seconds (did not complete) to optimize without
                    # calling evalf() at all, 19 seconds to optimize calling
                    # evalf() AFTER compose_u3, and 1 second to optimize
                    # calling evalf() BEFORE compose_u3.
                # 1. Here down, when we simplify, we add f(theta) to lambda to
                # correct the global phase when f(theta) is 2*pi. This isn't
                # necessary but the other steps preserve the global phase, so
                # we continue in that manner.
                # 2. The final step will remove Z rotations by 2*pi.
                # 3. Note that is_zero is true only if the expression is exactly
                # zero. If the input expressions have already been evaluated
                # then these final simplifications will not occur.
                # TODO After we refactor, we should have separate passes for
                # exact and approximate rewriting.

                # Y rotation is 0 mod 2*pi, so the gate is a u1
                if (right_parameters[0] % (2 * sympy.pi)).is_zero \
                        and right_name != "u1":
                    right_name = "u1"
                    right_parameters = (0, 0, right_parameters[1] +
                                        right_parameters[2] +
                                        right_parameters[0])
                # Y rotation is pi/2 or -pi/2 mod 2*pi, so the gate is a u2
                if right_name == "u3":
                    # theta = pi/2 + 2*k*pi
                    if ((right_parameters[0] - sympy.pi / 2) % (2 * sympy.pi)).is_zero:
                        right_name = "u2"
                        right_parameters = (sympy.pi / 2, right_parameters[1],
                                            right_parameters[2] +
                                            (right_parameters[0] - sympy.pi / 2))
                    # theta = -pi/2 + 2*k*pi
                    if ((right_parameters[0] + sympy.pi / 2) % (2 * sympy.pi)).is_zero:
                        right_name = "u2"
                        right_parameters = (sympy.pi / 2, right_parameters[1] +
                                            sympy.pi, right_parameters[2] -
                                            sympy.pi + (right_parameters[0] +
                                                        sympy.pi / 2))
                # u1 and lambda is 0 mod 2*pi so gate is nop (up to a global phase)
                if right_name == "u1" and (right_parameters[2] % (2 * sympy.pi)).is_zero:
                    right_name = "nop"
                # Simplify the symbolic parameters
                right_parameters = tuple(map(sympy.simplify, list(right_parameters)))
            # Replace the data of the first node in the run
            new_op = Instruction("", [], [], [])
            if right_name == "u1":
                new_op = U1Gate(right_parameters[2], run_qarg)
            if right_name == "u2":
                new_op = U2Gate(right_parameters[1], right_parameters[2], run_qarg)
            if right_name == "u3":
                new_op = U3Gate(*right_parameters, run_qarg)

            nx.set_node_attributes(dag.multi_graph, name='name',
                                   values={run[0]: right_name})
            nx.set_node_attributes(dag.multi_graph, name='op',
                                   values={run[0]: new_op})
            # Delete the other nodes in the run
            for current_node in run[1:]:
                dag._remove_op_node(current_node)
            if right_name == "nop":
                dag._remove_op_node(run[0])
        return dag

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
        quaternion_yzy = quaternion_from_euler([theta1, xi, theta2], 'yzy')
        euler = quaternion_yzy.to_zyz()
        quaternion_zyz = quaternion_from_euler(euler, 'zyz')
        # output order different than rotation order
        out_angles = (euler[1], euler[0], euler[2])
        abs_inner = abs(quaternion_zyz.data.dot(quaternion_yzy.data))
        if not np.allclose(abs_inner, 1, eps):
            raise MapperError('YZY and ZYZ angles do not give same rotation matrix.')
        out_angles = tuple(0 if np.abs(angle) < _CHOP_THRESHOLD else angle
                           for angle in out_angles)
        return out_angles
