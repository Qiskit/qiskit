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

"""Utility functions for routing"""

import numpy as np
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.layout import Layout
from qiskit.circuit import QuantumRegister, QuantumCircuit
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.controlflow.if_else import _unify_circuit_resources


def transpile_cf_multiblock(tpass, cf_node, current_layout, coupling, dag):
    """Transpile control flow instructions which may contain multiple
    blocks (e.g. IfElseOp). Since each control flow block may
    induce a yield a different layout, this function applies swaps
    to the shorter depth blocks to make all final layouts match.

    Args:
        tpass (BasePass): Transpiler pass object to use recursively.
        cf_op (IfElseOp): multiblock instruction.
        current_layout (Layout): The current layout at the start of the instruction.
        coupling (CouplingMap): the coupling map to use within the control flow instruction.
        qregs (list(QuantumRegister)): quantum registers for circuit

    Returns:
        IfElseOp: transpiled control flow operation
        final_layout (Layout): layout after instruction
    """
    # pylint: disable=cyclic-import
    from qiskit.transpiler.passes.routing.layout_transformation import LayoutTransformation
    from qiskit.converters import dag_to_circuit, circuit_to_dag

    block_circuits = []  # control flow circuit blocks
    block_dags = []  # control flow dag blocks
    block_layouts = []  # control flow layouts

    qregs = dag.qregs
    cf_op = cf_node.op
    for i, block in enumerate(cf_op.blocks):
        #block2 = copy_resources(resources, block)
        dag_block = circuit_to_dag(block)
        # if not cf_node.cargs and cf_node.cargs is not None:
        #     cf_node.cargs = None
        if len(dag_block.qubits) < len(dag.qubits):
            # expand block to full width
            #block2 = copy_resources(dag, block)
            dag_block2 = dag.copy_empty_like()
            if len(dag_block2.clbits) != len(dag_block.clbits):
                dag_block.clbits = dag_block2.clbits
            dag_block2.compose(dag_block, qubits=cf_node.qargs, clbits=cf_node.cargs)
            dag_block = dag_block2
        #dag_block.qregs = qregs
        try:
            updated_dag_block = tpass.run(dag_block)
        except TypeError as err:
            print(err)
            breakpoint()
            pass
        block_dags.append(updated_dag_block)
        block_layouts.append(tpass.property_set["final_layout"].copy())
    changed_layouts = [current_layout != layout for layout in block_layouts]

    if not any(changed_layouts):
        return cf_op, current_layout
    depth_cnt = [bdag.depth() for bdag in block_dags]
    maxind = np.argmax(depth_cnt)
    for i, dag in enumerate(block_dags):
        if i == maxind:
            block_circuits.append(dag_to_circuit(dag))
        else:
            layout_xform = LayoutTransformation(
                coupling,
                block_layouts[i],
                block_layouts[maxind],
            )
            match_dag = layout_xform.run(dag)
            block_circuits.append(dag_to_circuit(match_dag))
    final_permutation = combine_permutations(
        get_ordered_virtual_qubits(current_layout, qregs),
        get_ordered_virtual_qubits(block_layouts[maxind], qregs),
    )
    final_layout = Layout.from_intlist(final_permutation, *qregs.values())
    breakpoint()
    return cf_op.replace_blocks(block_circuits), final_layout


def transpile_cf_looping(tpass, cf_op, current_layout, coupling, qregs, dag):
    """For looping this pass adds a swap layer using LayoutTransformation
    to the end of the loop body to bring the layout back to the
    starting layout. This prevents reapplying layout changing
    swaps for every iteration of the loop.

    Args:
        tpass (BasePass): pass object to run
        cf_op (ForLoopOp, WhileLoopOp): looping instruction.
        current_layout (Layout): The current layout at the start and by the
           end of the instruction.
        coupling (CouplingMap): the coupling map to use within the control flow instruction.
        qregs (OrderedDict): QuantumRegister of top level circuit.

    Returns:
        tuple(ControlFlowOp, Layout): Transpiled control flow
            operation and layout after instruction

    """
    # pylint: disable=cyclic-import
    from qiskit.transpiler.passes.routing.layout_transformation import LayoutTransformation
    from qiskit.converters import dag_to_circuit, circuit_to_dag

    def _move_continue_to_end(dag):
        """Check if continue exists in block. If it does move it to the end of the circuit
        so layout_xform doesn't get skipped."""
        continue_nodes = dag.named_nodes("continue_loop")
        if continue_nodes:
            if len(continue_nodes) > 1:
                raise CircuitError("Multiple 'continue' statements contained in loop")
            if len(continue_nodes) <= 1:
                c_node = continue_nodes[-1]
                dag.remove_op_node(c_node)
                dag.apply_operation_back(c_node.op, c_node.qargs, c_node.cargs)
        return dag

    dag_block = circuit_to_dag(cf_op.blocks[0])
    dag_block.qregs = qregs
    start_qreg = QuantumRegister(coupling.size(), "q")
    start_layout = Layout.generate_trivial_layout(start_qreg)
    updated_dag_block = tpass.run(dag_block)
    updated_layout = tpass.property_set["final_layout"].copy()

    layout_xform = LayoutTransformation(coupling, updated_layout, start_layout)
    match_dag = layout_xform.run(updated_dag_block)
    match_dag = _move_continue_to_end(match_dag)

    match_circ = dag_to_circuit(match_dag)
    return cf_op.replace_blocks([match_circ]), current_layout


def get_ordered_virtual_qubits(layout, qregs):
    """Get list of virtual qubits associated with ordered list
    of physical qubits.

    Args:
        layout (Layout): circuit layout
        qregs (list(QuantumRegister)): list of registers for circuit

    Returns:
        list(int): list of virtual qubit indices
    """
    p2v = layout.get_virtual_bits()
    return [p2v[qubit] for qreg in qregs.values() for qubit in qreg]


def combine_permutations(*permutations):
    """
    Chain a series of permutations.

    Args:
        *permutations (list(int)): permutations to combine

    Returns:
        list: combined permutation
    """
    order = permutations[0]
    for this_order in permutations[1:]:
        order = [order[i] for i in this_order]
    return order


def layout_transform(cmap, layout, qreg=None):
    """Transform coupling map according to layout.

    Args:
       cmap (CouplingMap): coupling map to transform
       layout (Layout): layout to apply
       qreg (QuantumRegister): register to use for indexing

    Returns:
       CouplingMap: coupling map under specified layout.
    """
    if qreg is None:
        qreg = QuantumRegister(len(layout), "q")
    new_map = []
    vmap = layout.get_virtual_bits()
    for bit0, bit1 in cmap.get_edges():
        qubit0, qubit1 = qreg[bit0], qreg[bit1]
        new_map.append([vmap[qubit0], vmap[qubit1]])
    return CouplingMap(couplinglist=new_map)

def copy_resources(dag, qc):
    """
    One-way resource unification from dag to qc. It is 
    assumed that the resources of qc are a subset of the dag as
    for controlflow blocks.

    Args:
        dag (dict): source resources dictionary
        qc (QuantumCircuit): destination resources

    Returns:
        new QuantumCircuit instance
    """
    new_qc = QuantumCircuit()
    for reg in dag.qregs.values():
        new_qc.add_register(reg)
    for reg in dag.cregs.values():
        new_qc.add_register(reg)
    new_qc.compose(qc, qc.qubits, qc.clbits, inplace=True)
    return new_qc
