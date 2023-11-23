# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Helper function for converting a dag circuit to a dag dependency"""
from qiskit.dagcircuit.dagdependencyV2 import DAGDependencyV2


def dag_to_dagdependencyV2(dag):
    """Build a ``DAGDependencyV2`` object from a ``DAGCircuit``.

    Args:
        dag (DAGCircuit): the input dag.
        create_preds_and_succs (bool): whether to construct lists of
            predecessors and successors for every node.

    Return:
        DAGDependencyV2: the DAG representing the input circuit as a dag dependency.
    """

    dagdependency = DAGDependencyV2()
    dagdependency.name = dag.name
    dagdependency.metadata = dag.metadata

    dagdependency.add_qubits(dag.qubits)
    dagdependency.add_clbits(dag.clbits)

    for register in dag.qregs.values():
        dagdependency.add_qreg(register)

    for register in dag.cregs.values():
        dagdependency.add_creg(register)

    for node in dag.topological_op_nodes():
        # Get arguments for classical control (if any)
        inst = node.op.copy()
        dagdependency.add_op_node(inst, node.qargs, node.cargs)

    # copy metadata
    dagdependency.global_phase = dag.global_phase
    dagdependency.calibrations = dag.calibrations

    return dagdependency
