# -*- coding: utf-8 -*-

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

"""Helper function for converting a dag to a circuit."""
from qiskit.circuit import QuantumCircuit


def dag_to_circuit(dag):
    """Build a ``QuantumCircuit`` object from a ``DAGCircuit``.

    Args:
        dag (DAGCircuit): the input dag.

    Return:
        QuantumCircuit: the circuit representing the input dag.

    Example:
        .. jupyter-execute::

            from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
            from qiskit.dagcircuit import DAGCircuit
            from qiskit.converters import circuit_to_dag
            from qiskit.circuit.library.standard_gates import CHGate, U2Gate, CXGate
            from qiskit.converters import dag_to_circuit
            %matplotlib inline

            q = QuantumRegister(3, 'q')
            c = ClassicalRegister(3, 'c')
            circ = QuantumCircuit(q, c)
            circ.h(q[0])
            circ.cx(q[0], q[1])
            circ.measure(q[0], c[0])
            circ.rz(0.5, q[1]).c_if(c, 2)
            dag = circuit_to_dag(circ)
            circuit = dag_to_circuit(dag)
            circuit.draw()
    """

    name = dag.name or None
    circuit = QuantumCircuit(*dag.qregs.values(), *dag.cregs.values(), name=name,
                             global_phase=dag.global_phase)

    for node in dag.topological_op_nodes():
        # Get arguments for classical control (if any)
        inst = node.op.copy()
        inst.condition = node.condition
        circuit._append(inst, node.qargs, node.cargs)

    return circuit
