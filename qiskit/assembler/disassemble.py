# -*- coding: utf-8 -*-

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

"""Disassemble function for a qobj into a list of circuits and its config"""

import collections

from qiskit.circuit.classicalregister import ClassicalRegister
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister


def _experiments_to_circuits(qobj):
    """Return a list of QuantumCircuit object(s) from a qobj.

    Args:
        qobj (Qobj): The Qobj object to convert to QuantumCircuits

    Returns:
        list: A list of QuantumCircuit objects from the qobj
    """
    if qobj.experiments:
        circuits = []
        for x in qobj.experiments:
            quantum_registers = [QuantumRegister(i[1], name=i[0])
                                 for i in x.header.qreg_sizes]
            classical_registers = [ClassicalRegister(i[1], name=i[0])
                                   for i in x.header.creg_sizes]
            circuit = QuantumCircuit(*quantum_registers,
                                     *classical_registers,
                                     name=x.header.name)
            qreg_dict = collections.OrderedDict()
            creg_dict = collections.OrderedDict()
            for reg in quantum_registers:
                qreg_dict[reg.name] = reg
            for reg in classical_registers:
                creg_dict[reg.name] = reg
            conditional = {}
            for i in x.instructions:
                name = i.name
                if i.name == 'id':
                    name = 'iden'
                qubits = []
                params = getattr(i, 'params', [])
                try:
                    for qubit in i.qubits:
                        qubit_label = x.header.qubit_labels[qubit]
                        qubits.append(
                            qreg_dict[qubit_label[0]][qubit_label[1]])
                except Exception:  # pylint: disable=broad-except
                    pass
                clbits = []
                try:
                    for clbit in i.memory:
                        clbit_label = x.header.clbit_labels[clbit]
                        clbits.append(
                            creg_dict[clbit_label[0]][clbit_label[1]])
                except Exception:  # pylint: disable=broad-except
                    pass
                if hasattr(circuit, name):
                    instr_method = getattr(circuit, name)
                    if i.name in ['snapshot']:
                        _inst = instr_method(
                            i.label,
                            snapshot_type=i.snapshot_type,
                            qubits=qubits,
                            params=params)
                    elif i.name == 'initialize':
                        _inst = instr_method(params, qubits)
                    else:
                        _inst = instr_method(*params, *qubits, *clbits)
                elif name == 'bfunc':
                    conditional['value'] = int(i.val, 16)
                    full_bit_size = sum([creg_dict[x].size for x in creg_dict])
                    mask_map = {}
                    raw_map = {}
                    raw = []

                    for creg in creg_dict:
                        size = creg_dict[creg].size
                        reg_raw = [1] * size
                        if not raw:
                            raw = reg_raw
                        else:
                            for pos, val in enumerate(raw):
                                if val == 1:
                                    raw[pos] = 0
                            raw = reg_raw + raw
                        mask = [0] * (full_bit_size - len(raw)) + raw
                        raw_map[creg] = mask
                        mask_map[int("".join(str(x) for x in mask), 2)] = creg
                    creg = mask_map[int(i.mask, 16)]
                    conditional['register'] = creg_dict[creg]
                    val = int(i.val, 16)
                    mask = raw_map[creg]
                    for j in reversed(mask):
                        if j == 0:
                            val = val >> 1
                        else:
                            conditional['value'] = val
                            break
                else:
                    _inst = temp_opaque_instruction = Instruction(
                        name=name, num_qubits=len(qubits),
                        num_clbits=len(clbits), params=params)
                    circuit.append(temp_opaque_instruction, qubits, clbits)
                if conditional and name != 'bfunc':
                    _inst.c_if(conditional['register'], conditional['value'])
                    conditional = {}
            circuits.append(circuit)
        return circuits
    return None


def disassemble(qobj):
    """Disassemble a qobj and return the circuits, run_config, and user header.

    Args:
        qobj (Qobj): The input qobj object to disassemble

    Returns:
        tuple: (circuits, run_config, user_qobj_header):
            * circuits (list): A list of quantum circuits
            * run_config (dict): The dict of the run config
            * user_qobj_header (dict): The dict of any user headers in the qobj
    """
    run_config = qobj.config.to_dict()
    user_qobj_header = qobj.header.to_dict()
    circuits = _experiments_to_circuits(qobj)

    return circuits, run_config, user_qobj_header
