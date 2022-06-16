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

"""Map (with minimum effort) a DAGCircuit onto a `coupling_map` adding swap gates."""

import numpy as np
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.layout import Layout
from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.circuit import ControlFlowOp, IfElseOp, WhileLoopOp, ForLoopOp
from qiskit.converters import circuit_to_dag, dag_to_circuit


class BasicSwap(TransformationPass):
    """Map (with minimum effort) a DAGCircuit onto a `coupling_map` adding swap gates.

    The basic mapper is a minimum effort to insert swap gates to map the DAG onto
    a coupling map. When a cx is not in the coupling map possibilities, it inserts
    one or more swaps in front to make it compatible.
    """

    def __init__(self, coupling_map, fake_run=False, initial_layout=None):
        """BasicSwap initializer.

        Args:
            coupling_map (CouplingMap): Directed graph represented a coupling map.
            fake_run (bool): if true, it only pretend to do routing, i.e., no
                swap is effectively added.
            initial_layout (Layout, None): Initial layout of circuit.
        """
        super().__init__()
        self.coupling_map = coupling_map
        self.fake_run = fake_run
        self.property_set["final_layout"] = None
        self.initial_layout = initial_layout
        self._swap_layers = []  # used to store swap layers
        self._layout_layers = []  # used to store layout layers
        self._op_layers = []

    def run(self, dag):
        """Run the BasicSwap pass on `dag`.

        Args:
            dag (DAGCircuit): DAG to map.

        Returns:
            DAGCircuit: A mapped DAG.

        Raises:
            TranspilerError: if the coupling map or the layout are not
            compatible with the DAG.
        """
        if self.fake_run:
            return self.fake_run(dag)

        new_dag = dag.copy_empty_like()

        if len(dag.qregs) != 1 or dag.qregs.get("q", None) is None:
            raise TranspilerError("Basic swap runs on physical circuits only")

        if len(dag.qubits) > len(self.coupling_map.physical_qubits):
            raise TranspilerError("The layout does not match the amount of qubits in the DAG")

        canonical_register = dag.qregs["q"]
        if self.initial_layout:
            trivial_layout = self.initial_layout
        else:
            trivial_layout = Layout.generate_trivial_layout(canonical_register)
        current_layout = trivial_layout.copy()

        for layer in dag.serial_layers():
            subdag = layer["graph"]
            cf_layer = False
            # swap_layer = None
            for gate in subdag.two_qubit_ops():
                if isinstance(gate, ControlFlowOp):  # necessary?
                    continue
                physical_q0 = current_layout[gate.qargs[0]]
                physical_q1 = current_layout[gate.qargs[1]]
                if self.coupling_map.distance(physical_q0, physical_q1) != 1:
                    # Insert a new layer with the SWAP(s).
                    swap_layer = DAGCircuit()
                    swap_layer.add_qreg(canonical_register)

                    path = self.coupling_map.shortest_undirected_path(physical_q0, physical_q1)
                    for swap in range(len(path) - 2):
                        connected_wire_1 = path[swap]
                        connected_wire_2 = path[swap + 1]

                        qubit_1 = current_layout[connected_wire_1]
                        qubit_2 = current_layout[connected_wire_2]

                        # create the swap operation
                        swap_layer.apply_operation_back(
                            SwapGate(), qargs=[qubit_1, qubit_2], cargs=[]
                        )

                    # layer insertion
                    order = current_layout.reorder_bits(new_dag.qubits)
                    self._swap_layers.append(swap_layer)
                    new_dag.compose(swap_layer, qubits=order)

                    # update current_layout
                    for swap in range(len(path) - 2):
                        current_layout.swap(path[swap], path[swap + 1])
                    self._layout_layers.append(current_layout)
            # handle control flow operations
            for node in subdag.control_flow_ops():
                updated_ctrl_op, cf_layout = self._transpile_controlflow_op(node.op, current_layout)
                node.op = updated_ctrl_op
                cf_layer = True
            if cf_layer:
                new_dag.compose(subdag)
                current_layout = cf_layout
            else:
                order = current_layout.reorder_bits(new_dag.qubits)
                new_dag.compose(subdag, qubits=order)
            self._op_layers.append(subdag)
        self.property_set["final_layout"] = current_layout

        return new_dag

    def _transpile_controlflow_op(self, cf_op, current_layout):
        """default function"""
        if isinstance(cf_op, IfElseOp):
            return self._transpile_controlflow_multiblock(cf_op, current_layout)
        elif isinstance(cf_op, (ForLoopOp, WhileLoopOp)):
            return self._transpile_controlflow_looping(cf_op, current_layout)
        return cf_op, current_layout

    def _transpile_controlflow_multiblock(self, cf_op, current_layout):
        from qiskit.transpiler.passes.routing import LayoutTransformation

        block_circuits = []  # control flow circuit blocks
        block_dags = []  # control flow dag blocks
        block_layouts = []  # control flow layouts

        for i, block in enumerate(cf_op.blocks):
            dag_block = circuit_to_dag(block)
            _pass = BasicSwap(self.coupling_map, initial_layout=current_layout)
            updated_dag_block = _pass.run(dag_block)
            block_dags.append(updated_dag_block)
            block_layouts.append(_pass.property_set["final_layout"].copy())
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
                    self.coupling_map, block_layouts[i], block_layouts[maxind]
                )
                match_dag = layout_xform.run(block_dags[i])
                block_circuits.append(dag_to_circuit(match_dag))
        return cf_op.replace_blocks(block_circuits), block_layouts[maxind]

    def _transpile_controlflow_looping(self, cf_op, current_layout):
        """for looping this pass adds a swap layer at the end of the loop body to bring
        the layout back to the expected starting layout. This could be reduced a bit by
        specializing to for_loop and while_loop
        """
        from qiskit.transpiler.passes.routing import LayoutTransformation

        dag_block = circuit_to_dag(cf_op.blocks[0])
        _pass = BasicSwap(self.coupling_map, initial_layout=current_layout)
        updated_dag_block = _pass.run(dag_block)
        updated_layout = _pass.property_set["final_layout"].copy()
        layout_xform = LayoutTransformation(self.coupling_map, updated_layout, current_layout)
        match_dag = layout_xform.run(updated_dag_block)
        match_circ = dag_to_circuit(match_dag)
        return cf_op.replace_blocks([match_circ]), current_layout

    def _fake_run(self, dag):
        """Do a fake run the BasicSwap pass on `dag`.

        Args:
            dag (DAGCircuit): DAG to improve initial layout.

        Returns:
            DAGCircuit: The same DAG.

        Raises:
            TranspilerError: if the coupling map or the layout are not
            compatible with the DAG.
        """
        if len(dag.qregs) != 1 or dag.qregs.get("q", None) is None:
            raise TranspilerError("Basic swap runs on physical circuits only")

        if len(dag.qubits) > len(self.coupling_map.physical_qubits):
            raise TranspilerError("The layout does not match the amount of qubits in the DAG")

        canonical_register = dag.qregs["q"]
        trivial_layout = Layout.generate_trivial_layout(canonical_register)
        current_layout = trivial_layout.copy()

        for layer in dag.serial_layers():
            subdag = layer["graph"]
            for gate in subdag.two_qubit_ops():
                physical_q0 = current_layout[gate.qargs[0]]
                physical_q1 = current_layout[gate.qargs[1]]

                if self.coupling_map.distance(physical_q0, physical_q1) != 1:
                    path = self.coupling_map.shortest_undirected_path(physical_q0, physical_q1)
                    # update current_layout
                    for swap in range(len(path) - 2):
                        current_layout.swap(path[swap], path[swap + 1])

        self.property_set["final_layout"] = current_layout
        return dag
