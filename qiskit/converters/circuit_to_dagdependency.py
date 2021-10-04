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

"""Helper function for converting a circuit to a dag dependency"""

from qiskit.dagcircuit.dagdependency import DAGDependency


def circuit_to_dagdependency(circuit):
    """Build a ``DAGDependency`` object from a ``QuantumCircuit``.

    Args:
        circuit (QuantumCircuit): the input circuits.

    Return:
        DAGDependency: the DAG representing the input circuit as a dag dependency.
    """
    dagdependency = DAGDependency()
    dagdependency.name = circuit.name
    dagdependency.metadata = circuit.metadata

    dagdependency.add_qubits(circuit.qubits)
    dagdependency.add_clbits(circuit.clbits)

    for register in circuit.qregs:
        dagdependency.add_qreg(register)

    for register in circuit.cregs:
        dagdependency.add_creg(register)

    for operation, qargs, cargs in circuit.data:
        dagdependency.add_op_node(operation, qargs, cargs)

    dagdependency._add_successors()

    dagdependency.calibrations = circuit.calibrations

    return dagdependency
