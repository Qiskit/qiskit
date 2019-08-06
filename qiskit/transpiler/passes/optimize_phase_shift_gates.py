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

"""Pass for merging/canceling phase-shift gates like u1(theta), T-gates, S-gates and Z-gates
"""
import copy
import sys
import warnings
import numpy as np

from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.cx_cancellation import CXCancellation

np.set_printoptions(threshold=sys.maxsize)
warnings.filterwarnings('ignore')


class OptimizePhaseShiftGates(TransformationPass):
    """merge/cancel phase-shift gates in dag."""

    def run(self, dag):
        """
        Run one pass of phase-shift gate optimization and cx cancellation
        on the circuit. The function looks for the circuit configuration
        with the lowest number of gates and circuit depth

        Args:
            dag (DAGCircuit): the directed acyclic graph to run on.
        Returns:
            DAGCircuit: Transformed DAG.
        """
        t_counter = {}  # Counts the cumulative phase of T-gates and other z-rotation gates
        t_position_counter = {}  # Keeps track of the locations of T-gates and other z-rot. gates
        state_tracker = np.zeros((dag.width(), dag.size())).astype('int')
        for i in range(dag.width()):
            state_tracker[i][i] = 1  # initial state
        k = dag.width()  # number of initial variables
        for node in dag.topological_nodes():
            if node.type == 'op':
                if node.name == 'cx':
                    state_tracker[node.qargs[1][1]] ^= state_tracker[node.qargs[0][1]]
                elif node.name in ['t', 'tdg', 's', 'sdg', 'z']:
                    phase_t_gate_equivalent = {'t': 1, 'tdg': 7, 's': 2, 'sdg': 6, 'z': 4}[
                        node.name]
                    if str(state_tracker[node.qargs[0][1]]) in t_counter:
                        t_counter[str(state_tracker[node.qargs[0][1]])] += phase_t_gate_equivalent
                        t_position_counter[str(state_tracker[node.qargs[0][1]])].append(node)
                    else:
                        t_counter[str(state_tracker[node.qargs[0][1]])] = phase_t_gate_equivalent
                        t_position_counter[str(state_tracker[node.qargs[0][1]])] = [node]
                elif node.name == 'u1':
                    if str(state_tracker[node.qargs[0][1]]) in t_counter:
                        t_counter[str(state_tracker[node.qargs[0][1]])] += node.op.params[
                            0] * 4 / np.pi
                        t_position_counter[str(state_tracker[node.qargs[0][1]])].append(node)
                    else:
                        t_counter[str(state_tracker[node.qargs[0][1]])] = node.op.params[
                            0] * 4 / np.pi
                        t_position_counter[str(state_tracker[node.qargs[0][1]])] = [node]
                else:
                    state_tracker[node.qargs[0][1]] = np.zeros(dag.size())
                    state_tracker[node.qargs[0][1]][k] = 1
                    k += 1
        circuit_length_min = dag.node_counter+1
        circuit_depth_min = dag.depth() + 1
        cx_cancel_pass = CXCancellation()
        for key in t_counter:
            if t_counter[key] % 8 == 0:
                for i in t_position_counter[key]:
                    dag.remove_op_node(dag._id_to_node[i._node_id])  # Remove cancelled nodes
            else:
                for final_gate_position in range(len(t_position_counter[key])):
                    dag_copy = copy.deepcopy(dag)
                    k = 0
                    for i in t_position_counter[key]:
                        if k == final_gate_position:
                            p_reg = QuantumRegister(1, 'p')
                            repl = QuantumCircuit(p_reg)
                            if t_counter[key] % 8 == 1:
                                repl.t(p_reg[0])
                            elif t_counter[key] % 8 == 2:
                                repl.s(p_reg[0])
                            elif t_counter[key] % 8 == 4:
                                repl.z(p_reg[0])
                            elif t_counter[key] % 8 == 6:
                                repl.sdg(p_reg[0])
                            elif t_counter[key] % 8 == 7:
                                repl.tdg(p_reg[0])
                            else:
                                repl.u1((t_counter[key] % 8) * np.pi / 4, p_reg[0])
                            dag_repl = circuit_to_dag(repl)
                            dag_copy.substitute_node_with_dag(dag_copy._id_to_node[i._node_id],
                                                              dag_repl)
                        else:
                            dag_copy.remove_op_node(dag_copy._id_to_node[i._node_id])
                        k += 1
                    circuit_length = len(dag_copy._multi_graph.nodes)
                    circuit_depth = dag_copy.depth()
                    dag_copy = cx_cancel_pass.run(dag_copy)
                    if (circuit_length < circuit_length_min) or (
                            (circuit_length == circuit_length_min) and (
                                circuit_depth < circuit_depth_min)):
                        circuit_length_min = circuit_length
                        circuit_depth_min = circuit_depth
                        optimal_dag = copy.deepcopy(dag_copy)
                dag = optimal_dag
        return dag
