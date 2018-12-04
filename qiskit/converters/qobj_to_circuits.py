# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Helper function for converting qobj to a list of circuits"""

from qiskit import _classicalregister as cr
from qiskit import _quantumcircuit as qc
from qiskit import _quantumregister as qr

def _get_registers(reg_list, constructor):
    registers = {}
    for bits in reg_list:
        if bits[0] not in registers:
            registers[bits[0]] = [bits[1]]
        else:
            registers[bits[0]].append(bits[1])
    out_list = []
    for reg in registers:
        out_list.append(constructor(len(registers[reg]), name=reg))
    return out_list


def qobj_to_circuits(qobj):
    """Return a list of QuantumCircuit object(s) from a qobj

    Args:
        qobj (Qobj): The Qobj object to convert to QuantumCircuits
    Returns:
        list: A list of QuantumCircuit objects from the qobj

    """
    if qobj.experiments:
        circuits = []
        for x in qobj.experiments:
            if hasattr(x.header, 'compiled_circuit_qasm'):
                circuits.append(
                    qc.QuantumCircuit.from_qasm_str(
                        x.header.compiled_circuit_qasm))
            else:
                quantum_registers = _get_registers(x.header.qubit_labels,
                                                   qr.QuantumRegister)
                classical_registers = _get_registers(x.header.clbit_labels,
                                                     cr.ClassicalRegister)
                circuit = qc.QuantumCircuit(*quantum_registers,
                                            *classical_registers,
                                            name=x.header.name)
                qreg_dict = {}
                creg_dict = {}
                for reg in quantum_registers:
                    qreg_dict[reg.name] = reg
                for reg in classical_registers:
                    creg_dict[reg.name] = reg
                for i in x.instructions:
                    instr_method = getattr(circuit, i.name)
                    qubits = []
                    for qubit in i.qubits:
                        qubit_label = x.header.qubit_labels[qubit]
                        qubits.append(
                            qreg_dict[qubit_label[0]][qubit_label[1]])
                    clbits = []
                    for clbit in i.memory:
                        clbit_label = x.header.clbit_labels[clbit]
                        clbits.append(
                            creg_dict[clbit_label[0]][clbit_label[1]])
                    instr_method(*qubits, *clbits, *i.params)
                circuits.append(circuit)
        return circuits
    return None
