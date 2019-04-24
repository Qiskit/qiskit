# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=missing-return-type-doc

"""Disassemble function for a qobj into a list of circuits and it's config"""

from qiskit.circuit.classicalregister import ClassicalRegister
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister


# TODO: This is broken for conditionals. Will fix after circuits_2_qobj pr
def _experiments_to_circuits(qobj):
    """Return a list of QuantumCircuit object(s) from a qobj

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
            qreg_dict = {}
            creg_dict = {}
            for reg in quantum_registers:
                qreg_dict[reg.name] = reg
            for reg in classical_registers:
                creg_dict[reg.name] = reg
            for i in x.instructions:
                instr_method = getattr(circuit, i.name)
                qubits = []
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
                params = []
                try:
                    params = i.params
                except Exception:  # pylint: disable=broad-except
                    pass
                if i.name in ['snapshot']:
                    instr_method(*params)
                elif i.name == 'initialize':
                    instr_method(params, qubits)
                else:
                    instr_method(*params, *qubits, *clbits)
            circuits.append(circuit)
        return circuits
    return None


def disassemble(qobj):
    """Dissasemble a qobj and return the circuits, run_config, and user header

    Args:
        qobj (Qobj): The input qobj object to dissasemble
    Returns:
        circuits (list): A list of quantum circuits
        run_config (dict): The dist of the run config
        user_qobj_header (dict): The dict of any user headers in the qobj

    """
    run_config = qobj.config.to_dict()
    user_qobj_header = qobj.header.to_dict()
    circuits = _experiments_to_circuits(qobj)

    return circuits, run_config, user_qobj_header
