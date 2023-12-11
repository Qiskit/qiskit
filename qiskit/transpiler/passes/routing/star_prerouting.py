# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Search for star connectivity patterns and replace them with."""

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.circuit.library import SwapGate


class StarPreRouting(TransformationPass):
    """Run star to linear pre-routing

    This pass is a logical optimization pass that rewrites any
    solely 2q gate star connectivity subcircuit as a linear connectivity
    equivalent with swaps.

    For example:

      .. plot::
         :include-source:

         from qiskit.circuit import QuantumCircuit
         from qiskit.transpiler.passes import StarPreRouting

         qc = QuantumCircuit(10)
         qc.h(0)
         qc.cx(0, range(1, 5))
         qc.h(9)
         qc.cx(9, range(8, 4, -1))
         qc.measure_all()
         StarPreRouting()(qc).draw("mpl")

    This pass was inspired by a similar pass described in Section IV of:
    C. Campbell et al., "Superstaq: Deep Optimization of Quantum Programs,"
    2023 IEEE International Conference on Quantum Computing and Engineering (QCE),
    Bellevue, WA, USA, 2023, pp. 1020-1032, doi: 10.1109/QCE57702.2023.00116.
    """

    def run(self, dag):
        center_node = None
        star_sequences = []
        star_sequence = []
        for node in dag.topological_op_nodes():
            if (
                len(node.qargs) == 2
                and len(node.cargs) == 0
                and getattr(node.op, "condition", None) is None
            ):
                if center_node is None:
                    center_node = set(node.qargs)
                    star_sequence.append(node)
                elif isinstance(center_node, set):
                    if node.qargs[0] in center_node:
                        center_node = node.qargs[0]
                        star_sequence.append(node)
                    elif node.qargs[1] in center_node:
                        center_node = node.qargs[1]
                        star_sequence.append(node)
                    else:
                        center_node = None
                        star_sequence = []
                else:
                    if center_node in node.qargs:
                        star_sequence.append(node)
                    else:
                        saved_center = center_node
                        center_node = None
            elif len(node.qargs) > 2:
                saved_center = center_node
                center_node = None

            # If center_node is None after we've processed the node that
            # means we've broken star connectivity
            if center_node is None and len(star_sequence) > 1:
                star_sequences.append((star_sequence, saved_center))
                star_sequence = []
                if len(node.qargs) == 2:
                    center_node = set(node.qargs)
                    star_sequence.append(node)
        if len(star_sequence) > 1 and center_node:
            star_sequences.append((star_sequence, center_node))
        if not star_sequences:
            return dag
        new_dag = dag.copy_empty_like()
        sequence_starts = {x[0][0]: i for i, x in enumerate(star_sequences)}
        processed_nodes = set()
        qubit_mapping = {bit: index for index, bit in enumerate(dag.qubits)}
        for node in dag.topological_op_nodes():
            sequence_index = sequence_starts.get(node, None)
            if sequence_index is not None:
                index = sequence_index
                sequence, center_node = star_sequences[index]
                if len(sequence) == 2:
                    for node in sequence:
                        new_dag.apply_operation_back(
                            node.op,
                            _apply_mapping(node.qargs, qubit_mapping, dag.qubits),
                            node.cargs,
                        )
                        processed_nodes.add(node)
                    continue
                swap_source = None
                prev = None
                for inner_node in sequence:
                    if inner_node.qargs == prev:
                        new_dag.apply_operation_back(
                            inner_node.op,
                            _apply_mapping(inner_node.qargs, qubit_mapping, dag.qubits),
                            inner_node.cargs,
                        )
                        continue
                    if swap_source is None:
                        swap_source = center_node
                        new_dag.apply_operation_back(
                            inner_node.op,
                            _apply_mapping(inner_node.qargs, qubit_mapping, dag.qubits),
                            inner_node.cargs,
                        )
                        prev = inner_node.qargs
                        processed_nodes.add(inner_node)
                        continue
                    new_dag.apply_operation_back(
                        inner_node.op,
                        _apply_mapping(inner_node.qargs, qubit_mapping, dag.qubits),
                        inner_node.cargs,
                    )
                    processed_nodes.add(inner_node)
                    new_dag.apply_operation_back(
                        SwapGate(),
                        _apply_mapping(inner_node.qargs, qubit_mapping, dag.qubits),
                        inner_node.cargs,
                    )
                    # Swap mapping
                    pos_0 = qubit_mapping[inner_node.qargs[0]]
                    pos_1 = qubit_mapping[inner_node.qargs[1]]
                    qubit_mapping[inner_node.qargs[0]] = pos_1
                    qubit_mapping[inner_node.qargs[1]] = pos_0
                    prev = inner_node.qargs
            elif node in processed_nodes:
                continue
            else:
                new_dag.apply_operation_back(node.op, node.qargs, node.cargs)
        return new_dag


def _apply_mapping(qargs, mapping, qubits):
    return tuple(qubits[mapping[x]] for x in qargs)
