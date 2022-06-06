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

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.layout import Layout
from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.converters import circuit_to_dag, dag_to_circuit


class BasicSwap(TransformationPass):
    """Map (with minimum effort) a DAGCircuit onto a `coupling_map` adding swap gates.

    The basic mapper is a minimum effort to insert swap gates to map the DAG onto
    a coupling map. When a cx is not in the coupling map possibilities, it inserts
    one or more swaps in front to make it compatible.
    """

    def __init__(self, coupling_map, fake_run=False):
        """BasicSwap initializer.

        Args:
            coupling_map (CouplingMap): Directed graph represented a coupling map.
            fake_run (bool): if true, it only pretend to do routing, i.e., no
                swap is effectively added.
        """
        super().__init__()
        self.coupling_map = coupling_map
        self.fake_run = fake_run
        self.current_layout = None
        self._all_swaps = []

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
        if self.current_layout is None:
            trivial_layout = Layout.generate_trivial_layout(canonical_register)
            current_layout = trivial_layout.copy()
            self.current_layout = current_layout
        else:
            breakpoint()
            current_layout = self.current_layout
        #print(current_layout)
        for layer in dag.serial_layers():
            subdag = layer["graph"]
            print(dag_to_circuit(subdag))
            layer_swaps = []

            for gate in subdag.two_qubit_ops():
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
                    new_dag.compose(swap_layer, qubits=order)

                    # update current_layout
                    for swap in range(len(path) - 2):
                        current_layout.swap(path[swap], path[swap + 1])
                        #self._all_swaps.append((path[swap], path[swap + 1]))
                    self.current_layout = current_layout
                    print(dag_to_circuit(swap_layer))
            _apply_pass_controlflow(self, subdag)
            current_layout = self.current_layout
            order = current_layout.reorder_bits(new_dag.qubits)
            new_dag.compose(subdag, qubits=order)

        
        return new_dag

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


def _apply_pass_controlflow(_pass, dag):
    print('entering apply_pass_controlflow')
    for node in dag.control_flow_ops():
        flow_blocks = []
        block_layout = []
        starting_layout = _pass.current_layout.copy()
        breakpoint()
        print('starting layout\n', starting_layout)
        for i, block in enumerate(node.op.blocks):
            dag_block = circuit_to_dag(block)
            updated_dag_block = _pass.run(dag_block)
            block_layout.append(_pass.current_layout.copy())
            flow_circ_block = dag_to_circuit(updated_dag_block)
            flow_blocks.append(flow_circ_block)
            # reset current layout to track next block
            _pass.current_layout = starting_layout.copy()
            print(f'block layout {i}\n', block_layout, 'done')
        if len(block_layout) > 1:
            match_circ, match_layout = _layout_match(_pass, dag.num_qubits(), block_layout)
            breakpoint()
            flow_blocks[1].compose(match_circ, inplace=True)
        node.op = node.op.replace_blocks(flow_blocks)
        _pass.current_layout = match_layout
    print('exiting apply_pass_controlflow')            


def _layout_match(_pass, num_qubits, block_layout):
    from qiskit import QuantumCircuit
    from qiskit import transpile
    reference_layout = block_layout[0]  # should choose shortest circ as reference layout
    match_circs = []
    match_layouts = []
    print('reference block')
    print_ordered_layout(block_layout[0])    
    print('starting block 1')
    print_ordered_layout(block_layout[1])    
    for layout in block_layout[1:]:
        circ = QuantumCircuit(num_qubits)
        for qubit0, qubit1 in _pass._all_swaps[::-1]:
            circ.swap(qubit0, qubit1)
            layout.swap(qubit0, qubit1)
            print(qubit0, qubit1)
            print_ordered_layout(layout)
            print('-'*20)
        match_circs.append(circ)
        match_layouts.append(layout)
        # all match_layouts should match reference_layout
        print('final layout')
        print_ordered_layout(layout)
        breakpoint()
    try:
        assert(layout == reference_layout)
    except AssertionError:
        breakpoint()
    return circ, layout

def print_ordered_layout(layout):
    import collections
    import pprint
    od = collections.OrderedDict(sorted(layout._p2v.items()))
    pprint.pprint(od)
