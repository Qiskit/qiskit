# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Remove redundant Pauli product rotations before a Pauli product measurement."""

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.utils import control_flow

try:
    from qiskit._accelerate.remove_pauli_rotations_before_measure import (
        remove_pauli_rotations_before_measure,
    )
except (ImportError, AttributeError):
    remove_pauli_rotations_before_measure = None


def _pauli_products_match(dag, ppr_node, ppm_node):
    ppr = ppr_node.op
    ppm = ppm_node.op
    ppr_qargs = ppr_node.qargs
    ppm_qargs = ppm_node.qargs

    if len(ppr_qargs) != len(ppm_qargs):
        return False
    if len(ppr_qargs) != len(ppr._pauli_z) or len(ppm_qargs) != len(ppm._pauli_z):
        return False

    ppr_terms = sorted(
        (dag.find_bit(q).index, bool(z), bool(x)) for q, z, x in zip(ppr_qargs, ppr._pauli_z, ppr._pauli_x)
    )
    ppm_terms = sorted(
        (dag.find_bit(q).index, bool(z), bool(x)) for q, z, x in zip(ppm_qargs, ppm._pauli_z, ppm._pauli_x)
    )
    return ppr_terms == ppm_terms


def _remove_pauli_rotations_before_measure_py(dag):
    from qiskit.dagcircuit import DAGOpNode

    nodes_to_remove = []
    for node in dag.op_nodes(include_directives=True):
        if node.name != "pauli_product_measurement":
            continue

        ppm_node = node
        current_target = ppm_node
        while True:
            preds = list(dag.quantum_predecessors(current_target))
            if not preds:
                break
            first_pred = preds[0]
            if not all(p == first_pred for p in preds):
                break
            if not isinstance(first_pred, DAGOpNode):
                break
            if first_pred.name != "pauli_product_rotation":
                break

            succs = list(dag.quantum_successors(first_pred))
            if not all(s == current_target for s in succs):
                break

            if _pauli_products_match(dag, first_pred, ppm_node):
                nodes_to_remove.append(first_pred)
                current_target = first_pred
            else:
                break

    removed_nodes = set()
    for node in nodes_to_remove:
        if node not in removed_nodes:
            dag.remove_op_node(node)
            removed_nodes.add(node)


class RemovePauliRotationsBeforeMeasure(TransformationPass):
    """Remove redundant Pauli product rotations before a Pauli product measurement.

    Transpiler pass to remove :class:`~.PauliProductRotationGate` operations that
    immediately precede a :class:`~.PauliProductMeasurement` when the rotation's
    physical Pauli observable exactly matches the measurement's physical Pauli
    observable (up to the measurement sign).
    """

    @control_flow.trivial_recurse
    def run(self, dag):
        """Run the RemovePauliRotationsBeforeMeasure pass on `dag`.

        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: the optimized DAG.
        """
        if remove_pauli_rotations_before_measure is not None:
            remove_pauli_rotations_before_measure(dag)
        else:
            _remove_pauli_rotations_before_measure_py(dag)
        return dag
