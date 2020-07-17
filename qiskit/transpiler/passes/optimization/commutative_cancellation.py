# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Cancel the redundant (self-adjoint) gates through commutation relations."""

from collections import defaultdict
import numpy as np

from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.optimization.commutation_analysis import CommutationAnalysis
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit.library.standard_gates.u1 import U1Gate
from qiskit.circuit.library.standard_gates.rx import RXGate

_CUTOFF_PRECISION = 1E-5


class CommutativeCancellation(TransformationPass):
    """Cancel the redundant (self-adjoint) gates through commutation relations.

    Pass for cancelling self-inverse gates/rotations. The cancellation utilizes
    the commutation relations in the circuit. Gates considered include::

        H, X, Y, Z, CX, CY, CZ
    """

    def __init__(self):
        super().__init__()
        self.requires.append(CommutationAnalysis())

    def run(self, dag):
        """Run the CommutativeCancellation pass on `dag`.

        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: the optimized DAG.

        Raises:
            TranspilerError: when the 1-qubit rotation gates are not found
        """
        # Now the gates supported are hard-coded
        q_gate_list = ['cx', 'cy', 'cz', 'h', 'y']

        # Gate sets to be cancelled
        cancellation_sets = defaultdict(lambda: [])

        # Traverse each qubit to generate the cancel dictionaries
        # Cancel dictionaries:
        #  - For 1-qubit gates the key is (gate_type, qubit_id, commutation_set_id),
        #    the value is the list of gates that share the same gate type, qubit, commutation set.
        #  - For 2qbit gates the key: (gate_type, first_qbit, sec_qbit, first commutation_set_id,
        #    sec_commutation_set_id), the value is the list gates that share the same gate type,
        #    qubits and commutation sets.

        for wire in dag.wires:
            wire_name = "{0}[{1}]".format(str(wire.register.name), str(wire.index))
            wire_commutation_set = self.property_set['commutation_set'][wire_name]

            for com_set_idx, com_set in enumerate(wire_commutation_set):
                if com_set[0].type in ['in', 'out']:
                    continue
                for node in com_set:
                    num_qargs = len(node.qargs)
                    if num_qargs == 1 and node.name in q_gate_list:
                        cancellation_sets[(node.name, wire_name, com_set_idx)].append(node)
                    if num_qargs == 1 and node.name in ['z', 'u1', 'rz', 't', 's']:
                        cancellation_sets[('z_rotation', wire_name, com_set_idx)].append(node)
                    if num_qargs == 1 and node.name in ['rx', 'x']:
                        cancellation_sets[('x_rotation', wire_name, com_set_idx)].append(node)
                    # Don't deal with Y rotation, because Y rotation doesn't commute with CNOT, so
                    # it should be dealt with by optimized1qgate pass
                    elif num_qargs == 2 and node.qargs[0] == wire:
                        second_op_name = "{0}[{1}]".format(str(node.qargs[1].register.name),
                                                           str(node.qargs[1].index))
                        q2_key = (node.name, wire_name, second_op_name, com_set_idx,
                                  self.property_set['commutation_set'][(node, second_op_name)])
                        cancellation_sets[q2_key].append(node)

        for cancel_set_key in cancellation_sets:
            set_len = len(cancellation_sets[cancel_set_key])
            if set_len > 1 and cancel_set_key[0] in q_gate_list:
                gates_to_cancel = cancellation_sets[cancel_set_key]
                for c_node in gates_to_cancel[:(set_len // 2) * 2]:
                    dag.remove_op_node(c_node)

            elif set_len > 1 and cancel_set_key[0] in ['z_rotation', 'x_rotation']:
                run = cancellation_sets[cancel_set_key]
                run_qarg = run[0].qargs[0]
                total_angle = 0.0  # lambda
                for current_node in run:
                    if (current_node.condition is not None
                            or len(current_node.qargs) != 1
                            or current_node.qargs[0] != run_qarg):
                        raise TranspilerError("internal error")

                    if current_node.name in ['u1', 'rz', 'rx']:
                        current_angle = float(current_node.op.params[0])
                    elif current_node.name in ['z', 'x']:
                        current_angle = np.pi
                    elif current_node.name == 't':
                        current_angle = np.pi / 4
                    elif current_node.name == 's':
                        current_angle = np.pi / 2

                    # Compose gates
                    total_angle = current_angle + total_angle

                # Replace the data of the first node in the run
                if cancel_set_key[0] == 'z_rotation':
                    new_op = U1Gate(total_angle)
                elif cancel_set_key[0] == 'x_rotation':
                    new_op = RXGate(total_angle)

                if np.mod(total_angle, (2 * np.pi)) > _CUTOFF_PRECISION:
                    new_qarg = QuantumRegister(1, 'q')
                    new_dag = DAGCircuit()
                    new_dag.add_qreg(new_qarg)
                    new_dag.apply_operation_back(new_op, [new_qarg[0]])
                    dag.substitute_node_with_dag(run[0], new_dag)

                # Delete the other nodes in the run
                for current_node in run[1:]:
                    dag.remove_op_node(current_node)

                if np.mod(total_angle, (2 * np.pi)) < _CUTOFF_PRECISION:
                    dag.remove_op_node(run[0])

        return dag
