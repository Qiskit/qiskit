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

"""Helper function for converting a dag to a circuit"""
import collections

from qiskit.circuit import QuantumCircuit
from qiskit.circuit import ClassicalRegister
from qiskit.circuit import QuantumRegister


def dag_to_circuit(dag):
    """Build a ``QuantumCircuit`` object from a ``DAGCircuit``.

    Args:
        dag (DAGCircuit): the input dag.

    Return:
        QuantumCircuit: the circuit representing the input dag.
    """
    qregs = collections.OrderedDict()
    for qreg in dag.qregs.values():
        qreg_tmp = QuantumRegister(qreg.size, name=qreg.name)
        qregs[qreg.name] = qreg_tmp
    cregs = collections.OrderedDict()
    for creg in dag.cregs.values():
        creg_tmp = ClassicalRegister(creg.size, name=creg.name)
        cregs[creg.name] = creg_tmp

    name = dag.name or None
    circuit = QuantumCircuit(*qregs.values(), *cregs.values(), name=name)

    for node in dag.topological_op_nodes():
        qubits = []
        for qubit in node.qargs:
            qubits.append(qregs[qubit.register.name][qubit.index])

        clbits = []
        for clbit in node.cargs:
            clbits.append(cregs[clbit.register.name][clbit.index])

        # Get arguments for classical control (if any)
        inst = node.op.copy()
        inst.control = node.condition
        circuit.append(inst, qubits, clbits)
    return circuit
