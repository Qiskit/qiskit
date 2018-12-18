# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Helper function for converting a circuit to a dag"""


from qiskit.circuit.compositegate import CompositeGate
from qiskit.dagcircuit._dagcircuit import DAGCircuit


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
    # Add user gate definitions
    for name, data in circuit.definitions.items():
        dagcircuit.add_basis_element(name, data["n_bits"], 0, data["n_args"])
        dagcircuit.add_gate_data(name, data)
    # Add instructions
    builtins = {
        "U": ["U", 1, 0, 3],
        "CX": ["CX", 2, 0, 0],
        "measure": ["measure", 1, 1, 0],
        "reset": ["reset", 1, 0, 0],
        "barrier": ["barrier", -1, 0, 0]
    }
    # Add simulator instructions
    simulator_instructions = {
        "snapshot": ["snapshot", -1, 0, 1],
        "save": ["save", -1, 0, 1],
        "load": ["load", -1, 0, 1],
        "noise": ["noise", -1, 0, 1]
    }
    for main_instruction in circuit.data:
        # TODO: generate definitions and nodes for CompositeGates,
        # for now simply drop their instructions into the DAG
        instruction_list = []
        is_composite = isinstance(main_instruction, CompositeGate)
        if is_composite:
            instruction_list = main_instruction.instruction_list()
        else:
            instruction_list.append(main_instruction)

        for instruction in instruction_list:
            # Add OpenQASM built-in gates on demand
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

            def duplicate_instruction(inst):
                """Create a fresh instruction from an input instruction."""
                if inst.name == 'barrier':
                    params = [inst.qargs]
                elif inst.name in simulator_instructions.keys():
                    params = inst.param + [inst.qargs] + [inst.circuit]
                else:
                    params = inst.param + inst.qargs + inst.cargs
                new_inst = inst.__class__(*params)
                return new_inst

            inst = duplicate_instruction(instruction)
            dagcircuit.apply_operation_back(inst, inst.qargs,
                                            inst.cargs, control)

    return dagcircuit
