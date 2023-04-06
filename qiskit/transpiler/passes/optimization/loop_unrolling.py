# This code is part of Qiskit.
#
# (C) Copyright IBM 2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Loop unrolling optimizations"""

import numpy as np

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.utils import control_flow
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.circuit import IfElseOp, WhileLoopOp, ForLoopOp, ControlFlowOp, Instruction, Parameter
from qiskit.dagcircuit import DAGInNode, DAGOutNode
from qiskit.converters import dag_to_circuit, circuit_to_dag


class UnrollForLoops(TransformationPass):
    """Unroll for loops."""

    def run(self, dag):
        for node in dag.op_nodes(op=ForLoopOp):
            indexset, loop_parameter, body = node.op.params
            new_circ = body.copy_empty_like()
            if loop_parameter is None:
                for _ in range(len(indexset)):
                    new_circ.compose(body.copy(), inplace=True)
            elif isinstance(loop_parameter, Parameter):
                for index in range(len(indexset)):
                    i_body = body.assign_parameters({loop_parameter: index},
                                                    inplace=False)
                    new_circ.compose(i_body, inplace=True)
            new_dag = circuit_to_dag(new_circ)
            dag.substitute_node_with_dag(node, new_dag)
        return dag
    
    
class ForLoopBodyOptimizer(TransformationPass):
    """Optimize for loop bodies."""
            
    def __init__(self, opt_manager=None, pre_opt_cnt=0, inter_opt_cnt=0, post_opt_cnt=0):
        """
        Initialize for loop unroller. 

        Args: 
           opt_manager (None or PassManager): Attempt to optimize loop body across loop
             boundaries.
           pre_opt_cnt (None or int): how many operation back to check
           inter_opt_cnt (None or int): how deep to check from the front and back
           post_opt_cnt (None or int): how many operations to consider off the back
        """
        super().__init__()        
        self._opt = opt_manager
        self._pre_opt_cnt = pre_opt_cnt
        self._inter_opt_cnt = inter_opt_cnt
        self._post_opt_cnt = post_opt_cnt
        self._global_index_map = None

    def run(self, dag):
        self._global_index_map = {wire: idx for idx, wire in enumerate(dag.qubits)}
        for node in dag.op_nodes(op=ForLoopOp):
            # Optimize order here with pass manager?
            if self._pre_opt_cnt:
                self._optimize_pre(dag, node)
            if self._post_opt_cnt:
                self._optimize_post(dag, node)
        return dag

    def _optimize_pre(self, dag, node):
        indexset, loop_parameter, body = node.op.params
        trial_dag_in, block = _get_predecessor_dag(dag, node, self._pre_opt_cnt)
        #TODO: elliminate this conversion
        trial_dag_out = circuit_to_dag(self._opt.run([dag_to_circuit(trial_dag_in)])[0])
        if sum(trial_dag_out.count_ops().values()) < sum(trial_dag_in.count_ops().values()):
            # trim idle qubits so block qubits matches trial_dag_out
            trial_dag_out.remove_qubits(*trial_dag_out.idle_wires())
            print(dag_to_circuit(trial_dag_out))
            _replace_block_with_dag(dag, block, trial_dag_out)
            if len(indexset) > 1:
                new_for_loop = node.op.copy()
                new_for_loop.params = indexset[:-1], loop_parameter, body
                dag.substitute_node(node, new_for_loop)
            else:
                dag.remove_op_node(node)

    def _optimize_post(self, dag, node):
        indexset, loop_parameter, body = node.op.params
        trial_dag_in, block = _get_successor_dag(dag, node, self._post_opt_cnt)
        #TODO: elliminate this conversion
        trial_dag_out = circuit_to_dag(self._opt.run([dag_to_circuit(trial_dag_in)])[0])
        if sum(trial_dag_out.count_ops().values()) < sum(trial_dag_in.count_ops().values()):
            # trim idle qubits so block qubits matches trial_dag_out
            trial_dag_out.remove_qubits(*trial_dag_out.idle_wires())
            print(dag_to_circuit(trial_dag_out))
            _replace_block_with_dag(dag, block, trial_dag_out)
            if len(indexset) > 1:
                new_for_loop = node.op.copy()
                new_for_loop.params = indexset[:-1], loop_parameter, body
                dag.substitute_node(node, new_for_loop)
            else:
                dag.remove_op_node(node)
                
def _get_predecessor_dag(dag, node, cnt):
    """get the circuit formed from the body of the for-loop node and 
    previous cnt nodes"""
    indexset, loop_parameter, body = node.op.params

    # extract body of for_loop into new dag
    trial_dag = dag.copy_empty_like()
    trial_dag.global_phase = 0
    qubit_map = {body_qubit: dag_qubit for body_qubit, dag_qubit in zip(body.qubits, node.qargs)}
    clbit_map = {body_qubit: dag_qubit for body_clbit, dag_clbit in zip(body.clbits, node.cargs)}
    for circ_instr in body.data:
        qargs = [qubit_map[qubit] for qubit in circ_instr.qubits]
        cargs = [clbit_map[qubit] for clbit in circ_instr.clbits]        
        trial_dag.apply_operation_back(circ_instr.operation, qargs=qargs, cargs=cargs)
    # add predecessor nodes from dag
    pre_iter = dag.bfs_predecessors(node)
    block = [] # tracks the nodes added from _outside_ the loop
    while cnt > 0:
        print(f"cnt: {cnt} {node.name}")
        try:
            node_rec = next(pre_iter)
            print(node_rec)
        except StopIteration:
            print('stop iteration')
            breakpoint()
            break
        for child_node in node_rec[1]:
            print(repr(child_node))
            block.append(child_node)
            trial_dag.apply_operation_front(child_node.op, qargs=child_node.qargs, cargs=child_node.cargs)
            cnt -= 1
    return trial_dag, block

def _get_successor_dag(dag, node, cnt):
    """get the circuit formed from the body of the for-loop node and 
    subsequent cnt nodes"""
    indexset, loop_parameter, body = node.op.params

    # extract body of for_loop into new dag
    trial_dag = dag.copy_empty_like()
    trial_dag.global_phase = 0
    qubit_map = {body_qubit: dag_qubit for body_qubit, dag_qubit in zip(body.qubits, node.qargs)}
    clbit_map = {body_qubit: dag_qubit for body_clbit, dag_clbit in zip(body.clbits, node.cargs)}
    for circ_instr in body.data:
        qargs = [qubit_map[qubit] for qubit in circ_instr.qubits]
        cargs = [clbit_map[qubit] for clbit in circ_instr.clbits]        
        trial_dag.apply_operation_back(circ_instr.operation, qargs=qargs, cargs=cargs)

    # add predecessor nodes from dag
    post_iter = dag.bfs_successors(node)

    block = [] # tracks the nodes added from _outside_ the loop
    while cnt > 0:
        print(f"cnt: {cnt}")
        try:
            node_rec = next(post_iter)
        except StopIteration:
            print('stop iteration')
            breakpoint()
            break
        for child_node in node_rec[1]:
            print(repr(child_node))
            block.append(child_node)
            trial_dag.apply_operation_back(child_node.op, qargs=child_node.qargs, cargs=child_node.cargs)
            cnt -= 1
    return trial_dag, block

def _replace_block_with_dag(dag, block_a, dag_b):
    """
    Substitute a block of nodes with a dag
    There should be 1-to-1 correspondance between qubits in block_a and dag_b.
    """
    block_qubit_inds = sorted({dag.qubits.index(bit) for node in block_a for bit in node.qargs})
    if len(block_qubit_inds) != dag_b.num_qubits():
        breakpoint()
        raise TranspilerError(f"Number of qubits in block does not match replacement DAG")
    block_clbit_inds = sorted([dag.clbits.index(bit) for node in block_a for bit in node.cargs])
    index_map = {dag.qubits[ind]: ind for ind in block_qubit_inds}
    wire_map = {dag_b.qubits[index]: dag.qubits[qubit_ind]
                for index, qubit_ind in enumerate(block_qubit_inds)}
    placeholder_op = Instruction("placeholder", len(block_qubit_inds), len(block_clbit_inds), params=[])
    placeholder_node = dag.replace_block_with_op(
        block_a, placeholder_op, index_map
    )
    node_id_map = dag.substitute_node_with_dag(placeholder_node, dag_b)
    node_map = {node: node_id_map[node._node_id] for node in dag_b.op_nodes()}
    return node_map

def print_ids(qubits):
    for bit in qubits:
        print(hex(id(bit)), repr(bit))
