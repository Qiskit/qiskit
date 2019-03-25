# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Helper function for converting a circuit to a dag"""


from qiskit.circuit.compositegate import CompositeGate
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

    for main_instruction in circuit.data:
        # TODO: generate nodes for CompositeGates;
        # for now simply drop their instructions into the DAG
        instruction_list = []
        is_composite = isinstance(main_instruction, CompositeGate)
        if is_composite:
            instruction_list = main_instruction.instruction_list()
        else:
            instruction_list.append(main_instruction)

        for instruction in instruction_list:
            # Get arguments for classical control (if any)
            if instruction.control is None:
                control = None
            else:
                control = (instruction.control[0], instruction.control[1])

            def duplicate_instruction(inst):
                """Create a fresh instruction from an input instruction."""
                if inst.name == 'barrier':
                    params = [inst.qargs]
                elif inst.name == 'snapshot':
                    params = inst.params + [inst.qargs]
                else:
                    params = inst.params + inst.qargs + inst.cargs
                new_inst = inst.__class__(*params)
                return new_inst

            inst = duplicate_instruction(instruction)
            dagcircuit.apply_operation_back(inst, inst.qargs,
                                            inst.cargs, control)

    return dagcircuit
