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
from qiskit.transpiler.passes.optimization.axis_angle_analysis import AxisAngleAnalysis
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit.library.standard_gates.u1 import U1Gate
from qiskit.circuit.library.standard_gates.rx import RXGate
from qiskit.circuit.library.standard_gates.p import PhaseGate
from qiskit.circuit.library.standard_gates.rz import RZGate


_CUTOFF_PRECISION = 1E-5


class AxisAngleReduction(TransformationPass):
    """Reduce runs of single qubit gates with common axes.
    """

    def __init__(self):
        """
        AxisAngleReduction initializer.
        """
        super().__init__()
        if basis_gates:
            self.basis = set(basis_gates)
        else:
            self.basis = set()

        self._var_z_map = {'rz': RZGate, 'p': PhaseGate, 'u1': U1Gate}
        self.requires.append(AxisAngleAnalysis())

    def run(self, dag):
        """Run the AxisAngleReduction pass on `dag`.

        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: the optimized DAG.
        """

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
            wire_name = "{}[{}]".format(str(wire.register.name), str(wire.index))
            wire_commutation_set = self.property_set['commutation_set'][wire_name]

            for com_set_idx, com_set in enumerate(wire_commutation_set):
                if com_set[0].type in ['in', 'out']:
                    continue
                node_names = {node.name for node in com_set}
                for node in com_set:
                    num_qargs = len(node.qargs)
                    if num_qargs == 1 and node.name in self_inverse_gates:
                        cancellation_sets[(node.name, wire_name, com_set_idx)].append(node)
                    if num_qargs == 1 and node.name in std_z_gates:
                        cancellation_sets[('z_rotation', wire_name, com_set_idx)].append(node)
                    if num_qargs == 1 and node.name in std_x_gates:
                        cancellation_sets[('x_rotation', wire_name, com_set_idx)].append(node)
                    # Don't deal with Y rotation, because Y rotation doesn't commute with CNOT, so
                    # it should be dealt with by optimized1qgate pass
                    elif num_qargs == 2 and node.qargs[0] == wire:
                        second_op_name = "{}[{}]".format(str(node.qargs[1].register.name),
                                                         str(node.qargs[1].index))
                        q2_key = (node.name, wire_name, second_op_name, com_set_idx,
                                  self.property_set['commutation_set'][(node, second_op_name)])
                        cancellation_sets[q2_key].append(node)

        for cancel_set_key in cancellation_sets:
            if cancel_set_key[0] == 'z_rotation' and var_z_gate is None:
                continue
            set_len = len(cancellation_sets[cancel_set_key])
            if set_len > 1 and cancel_set_key[0] in self_inverse_gates:
                gates_to_cancel = cancellation_sets[cancel_set_key]
                for c_node in gates_to_cancel[:(set_len // 2) * 2]:
                    dag.remove_op_node(c_node)

            elif set_len > 1 and cancel_set_key[0] in ['z_rotation', 'x_rotation']:
                run = cancellation_sets[cancel_set_key]
                run_qarg = run[0].qargs[0]
                total_angle = 0.0  # lambda
                total_phase = 0.0
                for current_node in run:
                    if (current_node.condition is not None
                            or len(current_node.qargs) != 1
                            or current_node.qargs[0] != run_qarg):
                        raise TranspilerError("internal error")

                    if current_node.name in ['p', 'u1', 'rz', 'rx']:
                        current_angle = float(current_node.op.params[0])
                    elif current_node.name in ['z', 'x']:
                        current_angle = np.pi
                    elif current_node.name == 't':
                        current_angle = np.pi / 4
                    elif current_node.name == 's':
                        current_angle = np.pi / 2

                    # Compose gates
                    total_angle = current_angle + total_angle
                    if current_node.op.definition:
                        total_phase += current_node.op.definition.global_phase

                # Replace the data of the first node in the run
                if cancel_set_key[0] == 'z_rotation':
                    new_op = var_z_gate(total_angle)
                elif cancel_set_key[0] == 'x_rotation':
                    new_op = RXGate(total_angle)

                new_op_phase = 0
                if np.mod(total_angle, (2 * np.pi)) > _CUTOFF_PRECISION:
                    new_qarg = QuantumRegister(1, 'q')
                    new_dag = DAGCircuit()
                    new_dag.add_qreg(new_qarg)
                    new_dag.apply_operation_back(new_op, [new_qarg[0]])
                    dag.substitute_node_with_dag(run[0], new_dag)
                    if new_op.definition:
                        new_op_phase = new_op.definition.global_phase

                dag.global_phase = total_phase - new_op_phase

                # Delete the other nodes in the run
                for current_node in run[1:]:
                    dag.remove_op_node(current_node)

                if np.mod(total_angle, (2 * np.pi)) < _CUTOFF_PRECISION:
                    dag.remove_op_node(run[0])

        return dag


    def _commutation_analysis(self):
        
        if global_basis:
            #dfprop2 = dfprop.drop_duplicates(subset=['name', 'nparams', 'axis', 'angle'])
            dfaxis = dfprop.groupby('axis')
            buniq = dfprop.axis.unique()  # basis unique

            # merge collinear axes iff either contains a variable rotation
            for key, item in dfaxis:
                print(key)
                print(item)
                print('-'*20)
            print('unique axes')
            for i, v in enumerate(buniq):
                print(i, v)
            
            naxes = len(buniq)

            # 
            # index pairs of buniq which are vectors in opposite directions
            buniq_inverses = list()
            # for ind in range(naxes-1):
            #     for oind in range(ind, naxes):
            #         breakpoint()
            #         if math.isclose(math.abs(np.dot(buniq[ind], buniq[oind])), 1):
            #             buniq_inverses.append((ind, oind))
            breakpoint()
            for v1_ind, v1 in enumerate(buniq[:-1]):
                for v2_ind, v2 in enumerate(buniq[v1_ind+1:]):
                    if math.isclose(abs(v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]), 1):
                        print(np.dot(v1, v2), v1, v2)
                        buniq_inverses.append((v1_ind, v1_ind + 1 + v2_ind))
            print(buniq_inverses)
            breakpoint()
            #dfprop2 = dfprop.drop_duplicates(subset=['name', 'nparams', 'qubit', 'axis', 'angle'])
        else:
            raise CIrcuitError('not implemented')

                                             
            
        
