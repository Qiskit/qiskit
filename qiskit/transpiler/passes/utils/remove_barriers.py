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

"""Remove all barriers in a circuit"""

import logging
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.circuit.barrier import Barrier


class RemoveBarriers(TransformationPass):
    """Return a circuit with any barrier removed.


    For example, the circuit::

        qr = QuantumRegister(3, 'q')
        circuit = QuantumCircuit(qr)
        circuit.barrier(qr)

    Will be transformed into a circuit corresponding to::

        qr = QuantumRegister(3, 'q')

    """

    def run(self, dag):
        """Run the MergeAdjacentBarriers pass on `dag`."""

        # sorted to so that they are in the order they appear in the DAG
        # so ancestors/descendants makes sense
        barriers = [nd for nd in dag.topological_op_nodes() if nd.name == 'barrier']

        logging.info(f'RemoveBarriers: removing {barriers}')
        # add the merged barriers to a new DAG
        new_dag = dag._copy_circuit_metadata()

        # go over current nodes, and add them to the new dag
        for node in dag.topological_op_nodes():
            if node.name == 'barrier':
                pass
            else:
                # copy the condition over too
                new_dag.apply_operation_back(node.op, qargs=node.qargs, cargs=node.cargs)
        return new_dag
