# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Helper function for converting a circuit to a dag"""

from qiskit.circuit import Gate
from qiskit.circuit import Instruction
from qiskit.dagcircuit.dagcircuit import DAGCircuit


def circuit_to_dag(circuit):
    """Build a ``DAGCircuit`` object from a ``QuantumCircuit``.

    Args:
        circuit (QuantumCircuit): the input circuit.

    Return:
        DAGCircuit: the DAG representing the input circuit.
    """
    dagcircuit = DAGCircuit()
    dagcircuit.name = circuit.name
    for register in circuit.qregs:
        dagcircuit.add_qreg(register)
    for register in circuit.cregs:
        dagcircuit.add_creg(register)

    for instruction, qargs, cargs in circuit.data:
        # Get arguments for classical control (if any)
        if instruction.control is None:
            control = None
        else:
            control = (instruction.control[0], instruction.control[1])

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
                    new_inst = inst.__class__(inst.num_qubits, inst.num_clbits,
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

        dagcircuit.apply_operation_back(duplicate_instruction(instruction),
                                        qargs, cargs, control)

    return dagcircuit
