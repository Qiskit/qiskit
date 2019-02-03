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

        for instruction_context in instruction_list:
            # Add OpenQASM built-in gates on demand
            instruction = instruction_context[0]
            qargs = instruction_context[1]
            cargs = instruction_context[2]
            if instruction.name in builtins:
                dagcircuit.add_basis_element(*builtins[instruction.name])
            # Add simulator extension instructions
            if instruction.name in simulator_instructions:
                dagcircuit.add_basis_element(*simulator_instructions[instruction.name])
            # Get arguments for classical control (if any)
            if instruction.control is None:
                control = None
            else:
                control = (instruction.control[0], instruction.control[1])

            def duplicate_instruction(inst_ctx):
                """Create a fresh instruction from an input instruction."""
                inst = inst_ctx[0]
                qargs = inst_ctx[1]
                cargs = inst_ctx[2]
                args = inst.params + [inst.circuit]
                new_inst = inst.__class__(*args)
                return new_inst

            inst = duplicate_instruction(instruction_context)
            dagcircuit.apply_operation_back(inst, qargs, cargs, control)

    return dagcircuit
