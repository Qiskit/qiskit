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
from qiskit.circuit.library.standard_gates.p import PhaseGate
from qiskit.circuit.library.standard_gates.u import UGate
from qiskit.circuit.library.standard_gates.u1 import U1Gate
from qiskit.circuit.library.standard_gates.u2 import U2Gate
from qiskit.circuit.library.standard_gates.u3 import U3Gate
from qiskit.circuit.gate import Gate
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.quantum_info.synthesis import Quaternion

_CHOP_THRESHOLD = 1e-15


class Optimize1qGates(TransformationPass):
    """Optimize chains of single-qubit u1, u2, u3 gates by combining them into a single gate."""

    def __init__(self, basis=None, eps=1e-15):
        """Optimize1qGates initializer.

        Args:
            basis (list[str]): Basis gates to consider, e.g. `['u3', 'cx']`. For the effects
                of this pass, the basis is the set intersection between the `basis` parameter and
                the set `{'u1','u2','u3', 'u', 'p'}`.
            eps (float): EPS to check against
        """
        super().__init__()
        self.basis = basis if basis else ["u1", "u2", "u3"]
        self.eps = eps

    def run(self, dag):
        """Run the Optimize1qGates pass on `dag`.

        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: the optimized DAG.

        Raises:
            TranspilerError: if YZY and ZYZ angles do not give same rotation matrix.
        """
        use_u = "u" in self.basis
        use_p = "p" in self.basis
        runs = dag.collect_runs(["u1", "u2", "u3", "u", "p"])
        runs = _split_runs_on_parameters(runs)
        for run in runs:
            if use_p:
                right_name = "p"
            else:
                right_name = "u1"
            right_parameters = (0, 0, 0)  # (theta, phi, lambda)
            right_global_phase = 0
            for current_node in run:
                left_name = current_node.name
                if (
                    current_node.op.condition is not None
                    or len(current_node.qargs) != 1
                    or left_name not in ["p", "u1", "u2", "u3", "u", "id"]
                ):
                    raise TranspilerError("internal error")
                if left_name in ("u1", "p"):
                    left_parameters = (0, 0, current_node.op.params[0])
                elif left_name == "u2":
                    left_parameters = (
                        np.pi / 2,
                        current_node.op.params[0],
                        current_node.op.params[1],
                    )
                elif left_name in ("u3", "u"):
                    left_parameters = tuple(current_node.op.params)
                else:
                    if use_p:
                        left_name = "p"
                    else:
                        left_name = "u1"  # replace id with u1
                    left_parameters = (0, 0, 0)
                if (
                    current_node.op.definition is not None
                    and current_node.op.definition.global_phase
                ):
                    right_global_phase += current_node.op.definition.global_phase
                # If there are any sympy objects coming from the gate convert
                # to numpy.
                left_parameters = tuple(float(x) for x in left_parameters)
                # Compose gates
                name_tuple = (left_name, right_name)
                if name_tuple in (("u1", "u1"), ("p", "p")):
                    # u1(lambda1) * u1(lambda2) = u1(lambda1 + lambda2)
                    right_parameters = (0, 0, right_parameters[2] + left_parameters[2])
                elif name_tuple in (("u1", "u2"), ("p", "u2")):
                    # u1(lambda1) * u2(phi2, lambda2) = u2(phi2 + lambda1, lambda2)
                    right_parameters = (
                        np.pi / 2,
                        right_parameters[1] + left_parameters[2],
                        right_parameters[2],
                    )
                elif name_tuple in (("u2", "u1"), ("u2", "p")):
                    # u2(phi1, lambda1) * u1(lambda2) = u2(phi1, lambda1 + lambda2)
                    right_name = "u2"
                    right_parameters = (
                        np.pi / 2,
                        left_parameters[1],
                        right_parameters[2] + left_parameters[2],
                    )
                elif name_tuple in (("u1", "u3"), ("u1", "u"), ("p", "u3"), ("p", "u")):
                    # u1(lambda1) * u3(theta2, phi2, lambda2) =
                    #     u3(theta2, phi2 + lambda1, lambda2)
                    right_parameters = (
                        right_parameters[0],
                        right_parameters[1] + left_parameters[2],
                        right_parameters[2],
                    )
                elif name_tuple in (("u3", "u1"), ("u", "u1"), ("u3", "p"), ("u", "p")):
                    # u3(theta1, phi1, lambda1) * u1(lambda2) =
                    #     u3(theta1, phi1, lambda1 + lambda2)
                    if use_u:
                        right_name = "u"
                    else:
                        right_name = "u3"
                    right_parameters = (
                        left_parameters[0],
                        left_parameters[1],
                        right_parameters[2] + left_parameters[2],
                    )
                elif name_tuple == ("u2", "u2"):
                    # Using Ry(pi/2).Rz(2*lambda).Ry(pi/2) =
                    #    Rz(pi/2).Ry(pi-2*lambda).Rz(pi/2),
                    # u2(phi1, lambda1) * u2(phi2, lambda2) =
                    #    u3(pi - lambda1 - phi2, phi1 + pi/2, lambda2 + pi/2)
                    if use_u:
                        right_name = "u"
                    else:
                        right_name = "u3"
                    right_parameters = (
                        np.pi - left_parameters[2] - right_parameters[1],
                        left_parameters[1] + np.pi / 2,
                        right_parameters[2] + np.pi / 2,
                    )
                elif name_tuple[1] == "nop":
                    right_name = left_name
                    right_parameters = left_parameters
                else:
                    # For composing u3's or u2's with u3's, use
                    # u2(phi, lambda) = u3(pi/2, phi, lambda)
                    # together with the qiskit.mapper.compose_u3 method.
                    if use_u:
                        right_name = "u"
                    else:
                        right_name = "u3"
                    # Evaluate the symbolic expressions for efficiency
                    right_parameters = Optimize1qGates.compose_u3(
                        left_parameters[0],
                        left_parameters[1],
                        left_parameters[2],
                        right_parameters[0],
                        right_parameters[1],
                        right_parameters[2],
                    )
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
                if (
                    abs(np.mod(right_parameters[0], (2 * np.pi))) < self.eps
                    and right_name != "u1"
                    and right_name != "p"
                ):
                    if use_p:
                        right_name = "p"
                    else:
                        right_name = "u1"
                    right_parameters = (
                        0,
                        0,
                        right_parameters[1] + right_parameters[2] + right_parameters[0],
                    )
                # Y rotation is pi/2 or -pi/2 mod 2*pi, so the gate is a u2
                if right_name in ("u3", "u"):
                    # theta = pi/2 + 2*k*pi
                    right_angle = right_parameters[0] - np.pi / 2
                    if abs(right_angle) < self.eps:
                        right_angle = 0
                    if abs(np.mod((right_angle), 2 * np.pi)) < self.eps:
                        right_name = "u2"
                        right_parameters = (
                            np.pi / 2,
                            right_parameters[1],
                            right_parameters[2] + (right_parameters[0] - np.pi / 2),
                        )
                    # theta = -pi/2 + 2*k*pi
                    right_angle = right_parameters[0] + np.pi / 2
                    if abs(right_angle) < self.eps:
                        right_angle = 0
                    if abs(np.mod(right_angle, 2 * np.pi)) < self.eps:
                        right_name = "u2"
                        right_parameters = (
                            np.pi / 2,
                            right_parameters[1] + np.pi,
                            right_parameters[2] - np.pi + (right_parameters[0] + np.pi / 2),
                        )
                # u1 and lambda is 0 mod 2*pi so gate is nop (up to a global phase)
                if (
                    right_name in ("u1", "p")
                    and abs(np.mod(right_parameters[2], 2 * np.pi)) < self.eps
                ):
                    right_name = "nop"

            if right_name == "u2" and "u2" not in self.basis:
                if use_u:
                    right_name = "u"
                else:
                    right_name = "u3"
            if right_name in ("u1", "p") and right_name not in self.basis:
                if use_u:
                    right_name = "u"
                else:
                    right_name = "u3"

            new_op = Gate(name="", num_qubits=1, params=[])
            if right_name == "u1":
                new_op = U1Gate(right_parameters[2])
            if right_name == "p":
                new_op = PhaseGate(right_parameters[2])
            if right_name == "u2":
                new_op = U2Gate(right_parameters[1], right_parameters[2])
            if right_name == "u":
                if "u" in self.basis:
                    new_op = UGate(*right_parameters)
            if right_name == "u3":
                if "u3" in self.basis:
                    new_op = U3Gate(*right_parameters)
                else:
                    raise TranspilerError("It was not possible to use the basis %s" % self.basis)

            dag.global_phase += right_global_phase

            if right_name != "nop":
                dag.substitute_node(run[0], new_op, inplace=True)

            # Delete the other nodes in the run
            for current_node in run[1:]:
                dag.remove_op_node(current_node)
            if right_name == "nop":
                dag.remove_op_node(run[0])

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
        quaternion_yzy = Quaternion.from_euler([theta1, xi, theta2], "yzy")
        euler = quaternion_yzy.to_zyz()
        quaternion_zyz = Quaternion.from_euler(euler, "zyz")
        # output order different than rotation order
        out_angles = (euler[1], euler[0], euler[2])
        abs_inner = abs(quaternion_zyz.data.dot(quaternion_yzy.data))
        if not np.allclose(abs_inner, 1, eps):
            raise TranspilerError("YZY and ZYZ angles do not give same rotation matrix.")
        out_angles = tuple(0 if np.abs(angle) < _CHOP_THRESHOLD else angle for angle in out_angles)
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
