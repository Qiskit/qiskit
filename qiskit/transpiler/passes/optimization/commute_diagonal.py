# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Reduce CNOT count by using property that a unitary in SU4, which
generally requires three CNOT gates, can be decomposed into a unitary
which requires at most two CNOT gates plus a diagonal gate. For two
qubit unitaries which are seperated by something which commutes with
the diagonal, an overall reduction of CNOT gates can be obtained.
"""

import functools
import enum
import numpy as np
import cmath
import logging
import rustworkx
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes.utils import control_flow
from qiskit.converters import dag_to_circuit, circuit_to_dag
from qiskit.quantum_info import Operator
from qiskit.quantum_info.synthesis import two_qubit_decompose
from qiskit.extensions import UnitaryGate
from qiskit.circuit import Barrier, Gate, Qubit, Instruction
from qiskit.circuit.library.standard_gates import CXGate
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.exceptions import QiskitError

# BEGIN TEMPORARY DEBUG CODE
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
def cprint(text, bcolor, **keywords):
    print(bcolor + text + bcolors.ENDC, **keywords)

def print_node_ids(block):
    for node in block:
        print(node._node_id, node.name, hex(id(node)), node.op.params, end=' ')
        print(' '.join([hex(id(bit))for bit in node.qargs]))

def print_node_map(node_map):
    for node1, node2 in node_map.items():
        print(f'{node1._node_id} {node1.name} {hex(id(node1))} ==> {node2._node_id} {node2.name} {hex(id(node2))}')
        print(node1.op.params, node2.op.params, end=' ')
        print(' '.join([hex(id(bit))for bit in node1.qargs]), end=' ==> ')
        print(' '.join([hex(id(bit))for bit in node1.qargs]))        
    
def print_node_id_map(node_id_map):
    for key, node in node_id_map.items():
        print(f'{key}: {node._node_id} {node.name} {hex(id(node))} {node.op.params}')

def print_circuit_equal(op1, op2, local_phase=None):
    circuit_equal = op1 == op2
    color = bcolors.OKGREEN if circuit_equal == True else bcolors.FAIL
    cprint(f"DAG equals original? ", bcolors.OKBLUE, end='')
    cprint(str(circuit_equal), color)
    circuit_equiv = op1.equiv(op2)
    cprint(f"equivalent?: {circuit_equiv}", bcolors.OKBLUE)
    
    if not circuit_equal:
        if circuit_equiv:
            phase = np.angle(op1.data[0, 0]/op2.data[0, 0])
            cprint('phase offset = ', bcolors.OKBLUE, end='')
            cprint(f'{phase}', bcolors.WARNING)
            # attempt phase correction
            if local_phase:
                op1 *= np.exp(1j * (2*np.pi - local_phase) / 2)
                cprint(f'corrected equal? {op1 == op2}', bcolors.WARNING)
        breakpoint()

def get_phase_offset(op1, op2):
    return np.angle(op1.data[0, 0]/op2.data[0, 0])

def _candidate_block_printer(dag, candidate_blocks):
    qubit_to_int = {bit: idx for idx, bit in enumerate(dag.qubits)}
    for pair, block_list in candidate_blocks.items():
        bitinds = [qubit_to_int[qubit] for qubit in pair]
        print(f"bits: {bitinds}")
        prestr1 = " " * 3
        for i, block in enumerate(block_list):
            print(f"{prestr1}block {i}")
            prestr2 = prestr1 + " " * 3
            print(prestr2, end="")
            for node in block:
                print(f" {node.op.name}", end="")
            print("")

def print_blocks(pass_, dag):
    for i, block in enumerate(pass_._block_list):
        cprint(f'block {i}', bcolors.OKBLUE)
        print(_block_to_circuit(dag, block))

def print_block(dag, block):
    print(_block_to_circuit(dag, block))
    
# END TEMPORARY DEBUG CODE

class Diagonality(enum.Enum):
    """Class for defining 'how' diagonal an operation is"""

    # not diagonal
    FALSE = 0
    # diagonal
    TRUE = 1
    # conditionally diagonal using local 1q gates
    LOCAL = 2


class CommuteDiagonal(TransformationPass):
    """Decompose 2q unitaries with diagonals and push them to the right reducing overall CNOT gates"""

    def __init__(self):
        super().__init__()
        self._weyl_decomp = two_qubit_decompose.two_qubit_cnot_decompose
        self._diag_decomp = two_qubit_decompose.TwoQubitDecomposeUpToDiagonal()
        self._diag = np.identity(2, dtype=int)
        self._nondiag = np.ones((2, 2), dtype=int)
        self._global_index_map = None
        self._block_list = None

    @control_flow.trivial_recurse
    def run(self, dag):
        # identify 2-qubit runs
        self._global_index_map = {wire: idx for idx, wire in enumerate(dag.qubits)}
        blocks = dag.collect_2q_runs()
        candidate_blocks = self.get_candidate_2q_blocks(blocks, dag)
        # iterate over all 2q runs for each pair of qubits where they exist
        try:
            # this is just to track global phase of circuit: FIX
            oporig = Operator(dag_to_circuit(dag))
        except QiskitError:
            return dag
        for block_qubits, block_list in candidate_blocks.items():
            self._block_list = block_list
            if len(block_list) < 2:
                # we need at least 2 blocks to realize a reduction in cnot count
                continue
            num_blocks = len(block_list)
            # iterate over 2q runs for these two qubits
            for block_ind0, block_ind1 in zip(range(num_blocks - 1), range(1, num_blocks)):
                print(dag_to_circuit(dag))
                print_blocks(self, dag)
                print(f'block inds: ({block_ind0}, {block_ind1})')
                block0, block1 = block_list[block_ind0], block_list[block_ind1]
                inter_nodes = _collect_nodes_between_blocks(dag, block_qubits, block0, block1)
                if not inter_nodes:
                    # no nodes between blocks
                    continue
                if self._nodes_contain_barrier(inter_nodes):
                    # don't optimize over barrier
                    continue
                # check if blocks would benefit from pass
                circ0 = _block_to_circuit(dag, block0, remove_idle_qubits=True)
                circ1 = _block_to_circuit(dag, block1, remove_idle_qubits=True)
                cprint("circ0: ", bcolors.OKBLUE)
                print(circ0)
                cprint("circ1: ", bcolors.OKBLUE)
                print(circ1)
                mat0 = Operator(circ0).data
                mat1 = Operator(circ1).data
                num_cx = [self._weyl_decomp.num_basis_gates(mat) for mat in [mat0, mat1]]
                if not all((ncx == 3 for ncx in num_cx)):
                    # this block pair may not yield lower CNOT count
                    print(f'ncx = {num_cx}')
                    continue
                inter_dag = _copy_dag_bit_like(dag)
                for node in inter_nodes:
                    inter_dag.apply_operation_back(node.op, qargs=node.qargs, cargs=node.cargs)
                diagonality, diag_context = self.evaluate_diagonal_commutation(
                    inter_dag, block_qubits
                )
                print(diagonality)
                if diagonality == Diagonality.TRUE:
                    dmat, qc2cx = self._diag_decomp(mat0)
                    unitary0 = qc2cx.to_gate()
                    unitary1 = UnitaryGate(mat1 @ dmat)
                    self._replace_block_with_unitary(dag, block0, unitary0, block_qubits)
                    self._replace_block_with_unitary(dag, block1, unitary1, block_qubits)
                elif diagonality == Diagonality.LOCAL:
                    # the inter-block nodes are locally equivalent to diagonal
                    # add local equivalence ops
                    (
                        local_predeccessor_nodes,
                        nonlocal_node,
                        local_successor_nodes,
                        local_phase,
                    ) = diag_context
                    # add 1q ops of diag_context on block_qubits to block0 and block1
                    block0_local = block0 + [
                        node
                        for node in local_predeccessor_nodes
                        if set(node.qargs).issubset(set(block_qubits))
                    ]
                    block1_local = [
                        node
                        for node in local_successor_nodes
                        if set(node.qargs).issubset(set(block_qubits))
                    ] + block1

                    # group 1q ops of diag context NOT on block_qubits
                    inter_block_local0 = [
                        node
                        for node in local_predeccessor_nodes
                        if not set(node.qargs).issubset(set(block_qubits))
                    ]
                    inter_block_local1 = [
                        node
                        for node in local_successor_nodes
                        if not set(node.qargs).issubset(set(block_qubits))
                    ]
                    
                    circ0 = _block_to_circuit(dag, block0, remove_idle_qubits=True)
                    circ1 = _block_to_circuit(dag, block1, remove_idle_qubits=True)
                    dag0 = circuit_to_dag(circ0)
                    dag1 = circuit_to_dag(circ1)
                    circ0_2q = _block_to_circuit(dag0, block0_local, remove_idle_qubits=True)
                    circ1_2q = _block_to_circuit(dag1, block1_local, remove_idle_qubits=True)

                    mat0_2q = Operator(circ0_2q).data
                    mat1_2q = Operator(circ1_2q).data
                    dmat, qc2cx = self._diag_decomp(mat0_2q)
                    
                    mat_equal = np.allclose(mat0_2q, dmat @ Operator(qc2cx).data)
                    color = bcolors.OKGREEN if mat_equal else bcolors.FAIL                    
                    cprint(f"diag decomp phase correct ?", bcolors.OKBLUE, end="")
                    cprint(f"{mat_equal}", color)
                    if not mat_equal:
                        breakpoint()

                    from qiskit.extensions.quantum_initializer import DiagonalGate
                    diag_gate = DiagonalGate(np.diag(dmat).tolist())


                    inter_block_local = inter_block_local0 + [nonlocal_node] + inter_block_local1


                    print("starting dag\n", dag_to_circuit(dag))
                    print(_block_to_circuit(dag, block0))
                    print("section to replace\n", dag_to_circuit(inter_dag))
                    cprint(f"phase of new inter circuit: {local_phase}", bcolors.OKBLUE)
                    new_inter_nodes = local_predeccessor_nodes + [nonlocal_node] + local_successor_nodes

                    # UPDATE INTER BLOCK
                    node_map = self._replace_block_with_block(
                        dag,
                        inter_nodes,
                        new_inter_nodes)
                    print_blocks(self, dag)

                    # UPDATE LEFT BLOCK
                    block0_local = self._update_block_nodes(node_map, block0_local)
                    block1_local = self._update_block_nodes(node_map, block1_local)
                    qc2cx_dag = self._map_anonymous_circuit(qc2cx, dag0)
                    node_map = self._replace_block_with_dag(
                        dag, 
                        block0_local,
                        qc2cx_dag
                    )
                    print(dag_to_circuit(dag))
                    self._block_list[block_ind0] = qc2cx_dag.op_nodes()
                    self._update_block_lists_with_node(node_map)                    
                    print_blocks(self, dag)
                    #breakpoint()
                    
                    # UPDATE RIGHT BLOCK
                    block1_local_diag = [DAGOpNode(diag_gate, qargs=block_qubits)] + block1_local
                    block1_local_diag_circ = _block_to_circuit(dag, block1_local_diag,
                                                               remove_idle_qubits=True)
                    block1_local_diag_circ2 = self._weyl_decomp(Operator(block1_local_diag_circ).data)
                    block1_local_diag_dag = self._map_anonymous_circuit(block1_local_diag_circ2, dag1)

                    node_map = self._replace_block_with_dag(
                        dag,
                        block1_local,
                        block1_local_diag_dag)
                    self._block_list[block_ind1] = block1_local_diag_dag.op_nodes()
                    self._update_block_lists_with_node(node_map)
                    print_blocks(self, dag)
                    cprint(f'local_phase = {local_phase}', bcolors.OKBLUE)
                    if not cmath.isclose(np.sin(local_phase), 0, abs_tol=1e-9): 
                        #pass
                        phase_error = get_phase_offset(Operator(dag_to_circuit(dag)), oporig)
                        cprint(f'phase error = {phase_error}', bcolors.WARNING)
                        #breakpoint()
                        #if local_phase < np.pi:
                            
                        if (local_phase % (2 * np.pi)) > 3 * np.pi / 2:
                            # seems to need a different phase correction in the IV quadrant
                            dag.global_phase -= (2*np.pi - local_phase)
                        else:
                            dag.global_phase += local_phase
                    phase_error = get_phase_offset(oporig, Operator(dag_to_circuit(dag)))
                    blockop_orig = Operator(_block_to_circuit(dag, block0)) @ Operator(_block_to_circuit(dag, block1))
                    blockop_curr = Operator(_block_to_circuit(dag, self._block_list[block_ind0])) @ Operator(_block_to_circuit(dag, self._block_list[block_ind1]))
                    phase_error2 = get_phase_offset(blockop_orig, blockop_curr)
                    cprint(f'phase error (whole dag): {phase_error}', bcolors.WARNING)
                    cprint(f'phase error (blocks): {phase_error2}', bcolors.WARNING)            
                            
                    this_op = Operator(dag_to_circuit(dag))
                    cprint(f"evaluate final dag", bcolors.OKBLUE)
                    print_circuit_equal(this_op, oporig)
                    

                elif diagonality == Diagonality.FALSE:
                    continue
        return dag


    def _replace_block_with_unitary(self, dag, block, unitary, block_qubits):
        """
        Args:
           dag (DAGCircuit): parent dag of replacement
           block (List(Node)): nodes in dag to replace
           unitary (Instruction): unitary matrix instruction
           block_qubits (List(Qubit)): two qubit list of 2q block qubits.
        """
        block_index_map = _block_qargs_to_indices(block_qubits, self._global_index_map)
        new_node = dag.replace_block_with_op(block, unitary, block_index_map)
        # also need to update block_list
        self._block_list[self._block_list.index(block)] = [new_node]

    def _replace_block_with_block(self, dag, block_a, block_b, phase=None):
        """substitute a block of nodes with another set of nodes"""
        #block_index_map = _block_qargs_to_indices(block_qubits, self._global_index_map)        
        num_qubits = len({qubit for node in block_a for qubit in node.qargs})
        num_clbits = len({clbit for node in block_a for clbit in node.cargs})
        placeholder_op = Instruction('placeholder', num_qubits, num_clbits, params=[])
        placeholder_node = dag.replace_block_with_op(block_a, placeholder_op, self._global_index_map)
        dag_b = _block_to_dag(dag, block_b, remove_idle_qubits=True)
        dag_b_nodes = dag_b.op_nodes()
        cprint('replace_block_with_block', bcolors.OKBLUE)
        print_node_ids(block_b)
        print('--')
        print_node_ids(dag_b.op_nodes())
        print('--')            
        node_id_map = dag.substitute_node_with_dag(placeholder_node, dag_b)
        node_map = self._get_node_map(dag_b, node_id_map)
        # remap keys from dag_b to block_b
        new_node_map = {}
        for key, value in node_map.items():
            try:
                new_node_map[block_b[dag_b_nodes.index(key)]] = value
            except IndexError:
                print('uh oh')
                pass
        self._update_block_lists_with_node(new_node_map)
        if phase:
            dag.global_phase += phase
        return new_node_map

    def _replace_block_with_dag(self, dag, block_a, dag_b):
        """substitute a block of nodes with another set of nodes"""
        num_qubits = len({qubit for node in block_a for qubit in node.qargs})
        num_clbits = len({clbit for node in block_a for clbit in node.cargs})
        placeholder_op = Instruction('placeholder', num_qubits, num_clbits, params=[])
        placeholder_node = dag.replace_block_with_op(block_a, placeholder_op, self._global_index_map)
        node_id_map = dag.substitute_node_with_dag(placeholder_node, dag_b)
        
        #breakpoint()

        node_map = {node: node_id_map[node._node_id] for node in dag_b.op_nodes()}
        #self._update_block_lists_with_node(node_map)
        return node_map

    def _update_block_lists_with_id(self, node_id_map):
        for block in self._block_list:
            for node in block:
                if node._node_id in node_id_map:
                    node = node_id_map[node._node_id]

    def _update_block_lists_with_node(self, node_map):
        for i, block in enumerate(self._block_list):
            for j, node in enumerate(block):
                if node in node_map:
                    self._block_list[i][j] = node_map[node]
                    
    def _update_block_ids(self, block, node_id_map):
        for node in block:
            if node._node_id in node_id_map:
                node = node_id_map[node._node_id]

    def _map_anonymous_circuit(self, input_circ, ref_dag):
        dag_pre = circuit_to_dag(input_circ)
        new_dag = _copy_dag_bit_like(ref_dag)
        new_dag.compose(dag_pre)
        return new_dag
        

    def _get_node_map(self, input_dag, node_id_map):
        """
        Args:
           node_id_map(Dict): maps old node_id to new node
           input_dag (DAGCircuit): from circuit nodes which were replaced
        Return:
           dict: map old node to new node
        """
        return {node: node_id_map[node._node_id] for node in input_dag.op_nodes()
                if node._node_id in node_id_map.keys()}
        
    def _update_block_nodes(self, node_map, block):
        return [node_map[node] if node in node_map else node for node in block]
                    
    def get_candidate_2q_blocks(self, blocks, dag):
        """
        Return list of blocks which share same qubits.

        Args:
            blocks (List[DAGNode]): list of 2q DAGNode in topological order
            dag (DAGCircuit): reference dag

        Returns:
            List[DAGNode]: list of DAGNodes which are on common qubits.
        """
        common_blocks = {}
        for block in blocks:
            active_bits = set(qubit for node in block for qubit in node.qargs)
            key_2q = tuple(qubit for qubit in dag.qubits if qubit in active_bits)
            if key_2q in common_blocks:
                common_blocks[key_2q].append(block)
            else:
                common_blocks[key_2q] = [block]
        return common_blocks

    def evaluate_diagonal_commutation(self, inter_dag, partial_qubits, do_equiv_check=True):
        """
        Determine whether the circuit in dag commutes with a diagonal
        operation on a possible subset of its qubits.

        Args:
           inter_dag (DAGCircuit): circuit to check for commutation
           partial_qubits (List(Qubit)): qubits of dag to evaluate diagonal commutation on
           do_equiv_check (bool): whether to check for diagonal equivalence.

        Returns:
           Diagonality: type of diagonal for operation
           List(List(DAGNode), DAGNode, List(DAGNode)) or None: If Diagonality is LOCAL this
              contains the local predecessor nodes and local successor nodes of a non-local
              DAGNode which is diagonal on the partial qubits list.
        """
        is_diagonal = self._diagonal_commute_on_bits(dag_to_circuit(inter_dag), partial_qubits)
        if is_diagonal:
            return Diagonality.TRUE, None
        idle_wires = list(inter_dag.idle_wires())
        active_wires = [qubit for qubit in inter_dag.qubits if qubit not in idle_wires]
        num_active_qubits = len(active_wires)
        if do_equiv_check and not is_diagonal and num_active_qubits == 2:
            # although the group is not diagonal; check to see if it is locally equivalent
            # to a controlled gate.
            
            # create two-qubit circuit for two-qubit decomposer
            dag2q = self.copy_ops_like(inter_dag)
            qubit_map = {bit: index for index, bit in enumerate(dag2q.qubits)}
            dag2q.remove_qubits(*idle_wires)
            partial_qubits_2q = [qubit for qubit in dag2q.qubits if qubit in partial_qubits]
            mat2q = Operator(dag_to_circuit(dag2q)).data
            decomp = two_qubit_decompose.TwoQubitWeylDecomposition(mat2q)
            if not isinstance(decomp, two_qubit_decompose.TwoQubitWeylControlledEquiv):
                return Diagonality.FALSE, None
            # circuit is equivalent to controlled gate; use CX as controlled gate
            decomp_cx = two_qubit_decompose.TwoQubitBasisDecomposer(CXGate())
            circ_2q = decomp_cx(mat2q)
            if circ_2q.count_ops().get("cx", 0) != 1:
                return Diagonality.FALSE, None
            source_dag = circuit_to_dag(circ_2q)
            # want qubits like inter_dag but global_phase like circ_2q
            target_dag = _copy_dag_bit_like(inter_dag)
            #target_dag.global_phase = circ_2q.global_phase
            target_qubits = [
                target_dag.qubits[qubit_map[active_wires[ind]]]
                for ind in range(len(source_dag.qubits))
            ]
            target_dag.compose(source_dag, qubits=target_qubits)
            diag_context = self._get_local_pre_post_nodes(target_dag, "cx")
            nonlocal_node = diag_context[1]
            nonlocal_dag = _copy_dag_bit_like(target_dag)
            nonlocal_dag.apply_operation_back(
                nonlocal_node.op, qargs=nonlocal_node.qargs, cargs=nonlocal_node.cargs
            )
            cprint("equivalence inter-circuit check", bcolors.OKBLUE)
            cprint("starting matrix", bcolors.OKBLUE)
            print(mat2q)
            cprint('decomposition', bcolors.OKBLUE)
            print(circ_2q)
            print(Operator(circ_2q).data)
            circ_equal = Operator(mat2q) == Operator(circ_2q)
            color = bcolors.OKGREEN if circ_equal else bcolors.FAIL
            if not circ_equal:
                breakpoint()
            cprint(f'circuits equal? {circ_equal}', color)

            cprint('validate diag context: ', bcolors.OKBLUE, end='')
            (
                local_predeccessor_nodes,
                nonlocal_node,
                local_successor_nodes,
                local_phase,
            ) = diag_context

            diag_nodes = local_predeccessor_nodes + [nonlocal_node] + local_successor_nodes
            dag_nodes_circ = _block_to_circuit(inter_dag, diag_nodes)
            dag_nodes_circ.global_phase = local_phase
            
            if not cmath.isclose(np.sin(local_phase), 0, abs_tol=1e-8):
                print(circ_2q)
                print(dag_nodes_circ)
                #breakpoint()
            print_circuit_equal(Operator(dag_nodes_circ), Operator(dag_to_circuit(target_dag)))
            if Operator(dag_nodes_circ) != Operator(dag_to_circuit(target_dag)):
                print(dag_nodes_circ)
                print(dag_to_circuit(target_dag))
                #breakpoint()

            if self._diagonal_commute_on_bits(dag_to_circuit(nonlocal_dag), partial_qubits_2q):
                return Diagonality.LOCAL, diag_context
            else:
                return Diagonality.FALSE, None
        return Diagonality.FALSE, None

    def _diagonal_commute_on_bits(self, circuit, partial_qubits):
        """
        Returns whether circuit is diagonal on specified qubits.

        Args:
            circuit (QuantumCircuit): circuit to check for diagonal commutation
            partial_qubits (List(Qubit)): list of qubits of circuit to check for diagonal commutation.

        Returns:
            bool: whether circuit commutes with diagonal operation on selected qubits.
        """
        mat = Operator(circuit).data
        components = [
            self._diag if qubit in partial_qubits else self._nondiag for qubit in circuit.qubits
        ]
        pattern = functools.reduce(np.kron, components[::-1]).astype(int)
        zero_mask = np.nonzero(pattern == 0)
        return np.allclose(mat[zero_mask], 0)

    def _get_local_pre_post_nodes(self, target_dag, node_name):
        """
        Split off the pre and post 1q gates of the single 2q gate.

        Args:
            target_dag (DAGCircuit): circuit from two_qubit_cnot_decompose
            node_name (str): name of nonlocal node

        Returns:
            List(DAGNode): 1q nodes before 2q node
            DAGNode: 2q node
            List(DAGNode): 1q nodes after 2q node
            float: phase angle of this decomposition

        Raises:
            TranspilerError: target_dag does not have exactly one node named node_name
        """
        nonlocal_nodes = target_dag.named_nodes(node_name)
        if len(nonlocal_nodes) != 1:
            raise TranspilerError("expected exactly one CNOT gate in circuit")
        nonlocal_node = nonlocal_nodes[0]
        pred_nodes = [
            node
            for node in target_dag.quantum_predecessors(nonlocal_node)
            if isinstance(node, DAGOpNode)
        ]
        succ_nodes = [
            node
            for node in target_dag.quantum_successors(nonlocal_node)
            if isinstance(node, DAGOpNode)
        ]
        return pred_nodes, nonlocal_node, succ_nodes, target_dag.global_phase

    def copy_ops_like(self, dag):
        """Return a copy of dag with same qubit and clbit instances but copies of operations

        This is like copy_empty_like but includes a copy of the operations from this dag. Also
        copies anything 'copy_empty_like' copies, namely global_phase.

        Args:
            dag (DAGCircuit): circuit to copy

        Returns:
            DAGCircuit: copy of dag
        """
        new_dag = dag.copy_empty_like()
        for node in dag.op_nodes():
            new_dag.apply_operation_back(node.op.copy(), qargs=node.qargs, cargs=node.cargs)
        return new_dag

    def _nodes_contain_barrier(self, nodes):
        for node in nodes:
            if isinstance(node, DAGOpNode) and isinstance(node.op, Barrier):
                return True
        return False


def _collect_nodes_between_blocks(dag, block_qubits, block0, block1):
    """collect nodes between 2q blocks"""

    _, start_node0 = _get_first_last_node(dag, block_qubits[0], block0)
    _, start_node1 = _get_first_last_node(dag, block_qubits[1], block0)
    if start_node0 is None and start_node1 is None:
        return []
    stop_node0, _ = _get_first_last_node(dag, block_qubits[0], block1)
    stop_node1, _ = _get_first_last_node(dag, block_qubits[1], block1)
    if stop_node0 is None and stop_node1 is None:
        return []
    inter_nodes = _collect_circuit_between_nodes(
        dag, block_qubits, (start_node0, start_node1), (stop_node0, stop_node1)
    )
    if len(inter_nodes) == 0:
        breakpoint()
    return inter_nodes

def _get_first_node(dag, wire, block):
    """get first node of wire in block"""
    wire_iter = dag.nodes_on_wire(wire)
    node = next(wire_iter)
    while node not in block:
        try:
            node = next(wire_iter)
        except StopIteration:
            return None
    return node

def _get_first_last_node(dag, wire, block):
    """get first and last node which are in the block and on the wire"""

    wire_iter = dag.nodes_on_wire(wire)
    # iterate to first op node of block on wire
    node = next(wire_iter)
    while node not in block:
        try:
            node = next(wire_iter)
        except StopIteration:
            # no nodes on wire exist in block
            return None, None
    first_node = node
    # iterate to last op node
    next_node = next(wire_iter)
    while next_node in block:
        node = next_node
        next_node = next(wire_iter)
    last_node = node
    return first_node, last_node


def _collect_circuit_between_nodes(dag, block_qubits, start_nodes, stop_nodes):
    """collect circuit between 2q nodes"""
    start_node0, start_node1 = start_nodes
    iter_q0 = dag.nodes_on_wire(block_qubits[0], only_ops=False)  # False to allow DAGOutNode
    iter_q1 = dag.nodes_on_wire(block_qubits[1], only_ops=False)
    # advance iterators to start node
    _advance_iterator_to_node(iter_q0, start_node0)
    _advance_iterator_to_node(iter_q1, start_node1)
    inter_nodes = _gather_to_stop_nodes(block_qubits, iter_q0, iter_q1, stop_nodes)
    return inter_nodes


def _advance_iterator_to_node(iterator, stop_node, gathered=None):
    """
    Args:
        iterator (iterator):
        stop_node (DAGNode): node to iterate to
        gathered (None or List(Node)): If list, accumulate nodes into this list

    Returns:
        iterator: node iterator pointing to stop_node
    """
    node = next(iterator)
    while node != stop_node:
        if gathered is not None:
            gathered.append(node)
        node = next(iterator)
    if gathered is not None and node != stop_node:
        gathered.append(node)
    return node


def _gather_to_stop_nodes(block_qubits, iter_q0, iter_q1, stop_nodes):
    """
    Gathers nodes interacting with two qubits until stop nodes are encountered.

    Works by iterating on wire0 until the stop node is encountered or there is a joint operation,
    in which case wire1 is accumulated until it reaches the same joint operation, then
    accumulation continues with wire0 leading the way.
    """
    stop_node0, stop_node1 = stop_nodes
    gathered = []
    node0 = next(iter_q0)
    while node0 != stop_node0:
        node0_qubits = set(node0.qargs) if hasattr(node0, "qargs") else set([node0.wire])
        if set(block_qubits).issubset(node0_qubits):
            # operation interacts with q1; accumulate on q1
            node1 = _advance_iterator_to_node(iter_q1, node0, gathered=gathered)
            gathered.append(node0)  # same as node1
            node0 = next(iter_q0)
        else:
            gathered.append(node0)
            node0 = next(iter_q0)
    # catch up iter_q1 if needed
    try:
        node1 = next(iter_q1)
    except StopIteration:
        pass
    else:
        if node1 != stop_node1:
            gathered.append(node1)
            _advance_iterator_to_node(iter_q1, stop_node1, gathered=gathered)
    return gathered


def _block_qargs_to_indices(block_qargs, global_index_map):
    """Map each qubit in block_qargs to its wire position among the block's wires.

    This code is taken from ConsolidateBlocks.

    Args:
        block_qargs (list): list of qubits that a block acts on
        global_index_map (dict): mapping from each qubit in the
            circuit to its wire position within that circuit
    Returns:
        dict: mapping from qarg to position in block
    """
    block_indices = [global_index_map[q] for q in block_qargs]
    ordered_block_indices = {bit: index for index, bit in enumerate(sorted(block_indices))}
    block_positions = {q: ordered_block_indices[global_index_map[q]] for q in block_qargs}
    return block_positions


def _block_to_dag(dag, block, remove_idle_qubits=False):
    block_dag = _copy_dag_bit_like(dag)
    for node in block:
        block_dag.apply_operation_back(node.op, qargs=node.qargs, cargs=node.cargs)
    if remove_idle_qubits:
        block_dag.remove_qubits(*{bit for bit in block_dag.idle_wires() if isinstance(bit, Qubit)})
    return block_dag

def _block_to_circuit(dag, block, remove_idle_qubits=False):
    block_dag = _block_to_dag(dag, block, remove_idle_qubits=remove_idle_qubits) 
    return dag_to_circuit(block_dag)

def _copy_dag_bit_like(dag):
    target_dag = DAGCircuit()
    target_dag.add_qubits(dag.qubits)
    target_dag.add_clbits(dag.clbits)

    for qreg in dag.qregs.values():
        target_dag.add_qreg(qreg)
    for creg in dag.cregs.values():
        target_dag.add_creg(creg)
    return target_dag


def collect_2q_runs_on_qubits(dag, run_qubits):
    """Return a set of non-conditional runs of "op" nodes which directly connect to run_qubits."""
    to_qid = {}
    run_qubits_set = set(run_qubits)
    for i, qubit in enumerate(dag.qubits):
        to_qid[qubit] = i

    def filter_fn(node):
        if isinstance(node, DAGOpNode):
            return (
                isinstance(node.op, Gate)
                and len(node.qargs) <= 2
                and not getattr(node.op, "condition", None)
                and not node.op.is_parameterized()
                and set(node.qargs).issubset(run_qubits_set)
            )
        else:
            return None

    def color_fn(edge):
        if isinstance(edge, Qubit):
            return to_qid[edge]
        else:
            return None

    return rustworkx.collect_bicolor_runs(dag._multi_graph, filter_fn, color_fn)
