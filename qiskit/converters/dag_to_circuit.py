# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Helper function for converting a dag to a circuit"""
import collections

from qiskit.circuit import ClassicalRegister
from qiskit.circuit import Instruction
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
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

    for node in dag.nodes_in_topological_order():
        if node.type == 'op':
            qubits = []
            for qubit in node.qargs:
                qubits.append(qregs[qubit[0].name][qubit[1]])

            clbits = []
            for clbit in node.cargs:
                clbits.append(cregs[clbit[0].name][clbit[1]])

            # Get arguments for classical control (if any)
            if node.condition is None:
                control = None
            else:
                control = (node.condition[0], node.condition[1])

            def duplicate_instruction(inst):
                """Create a fresh instruction from an input instruction."""
                if issubclass(inst.__class__,
                              Instruction) and inst.__class__ not in [
                                  Instruction, Gate]:
                    if inst.name == 'barrier':
                        new_inst = inst.__class__(inst.num_qubits)
                    elif inst.name == 'initialize':
                        params = getattr(inst, 'params', [])
                        new_inst = inst.__class__(params)
                    elif inst.name == 'snapshot':
                        label = inst.params[0]
                        snap_type = inst.params[1]
                        new_inst = inst.__class__(inst.num_qubits,
                                                  inst.num_clbits,
                                                  label, snap_type)
                    else:
                        params = getattr(inst, 'params', [])
                        new_inst = inst.__class__(*params)
                else:
                    new_inst = Instruction(name=inst.name,
                                           num_qubits=inst.num_qubits,
                                           num_clbits=inst.num_clbits,
                                           params=inst.params)
                    new_inst.definition = inst.definition
                return new_inst

            inst = duplicate_instruction(node.op)
            inst.control = control
            circuit.append(inst, qubits, clbits)

    return circuit
