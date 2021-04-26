# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Support via qiskit.qasm2 for functional interface
to Qasm2 source loading and dumping in functions.py
"""

import warnings
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.gate import Gate
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.qasm2 import Qasm, QasmError


def qasm_load(qasm: Qasm) -> QuantumCircuit:
    """
    Factory OpenQASM src into QuantumCircuit
    using qiskit.qasm code.

    Parameters
    ----------
    qasm : Qasm
        The Qasm object of source to load.

    Returns
    -------
    QuantumCircuit
        The resulting QuantumCircuit.

    """
    from qiskit.converters import ast_to_dag, dag_to_circuit
    ast = qasm.parse()
    dag = ast_to_dag(ast)
    return dag_to_circuit(dag)


def qasm_dump(qc: QuantumCircuit) -> str:
    """
    Return OpenQASM string using qiskit.qasm code.

    Parameters
    ----------
    qc : QuantumCircuit
        The circuit to be disassembled into OpenQASM source

    Raises
    ------
    QasmError
        If circuits have unbound parameters

    Returns
    -------
    str
        OpenQASM source for the circuit.

    """

    if qc.num_parameters > 0:
        raise QasmError('Cannot represent circuits with unbound parameters in OpenQASM 2.')

    existing_gate_names = ['ch', 'cp', 'cx', 'cy', 'cz', 'crx', 'cry', 'crz', 'ccx', 'cswap',
                           'csx', 'cu', 'cu1', 'cu3', 'dcx', 'h', 'i', 'id', 'iden', 'iswap',
                           'ms', 'p', 'r', 'rx', 'rxx', 'ry', 'ryy', 'rz', 'rzx', 'rzz', 's',
                           'sdg', 'swap', 'sx', 'x', 'y', 'z', 't', 'tdg', 'u', 'u1', 'u2',
                           'u3']

    existing_composite_circuits = []

    string_temp = qc.header + "\n"
    string_temp += qc.extension_lib + "\n"
    for register in qc.qregs:
        string_temp += register.qasm() + "\n"
    for register in qc.cregs:
        string_temp += register.qasm() + "\n"

    qreg_bits = set(bit for reg in qc.qregs for bit in reg)
    creg_bits = set(bit for reg in qc.cregs for bit in reg)
    regless_qubits = []
    regless_clbits = []

    if set(qc.qubits) != qreg_bits:
        regless_qubits = [bit for bit in qc.qubits if bit not in qreg_bits]
        string_temp += "qreg %s[%d];\n" % ('regless', len(regless_qubits))

    if set(qc.clbits) != creg_bits:
        regless_clbits = [bit for bit in qc.clbits if bit not in creg_bits]
        string_temp += "creg %s[%d];\n" % ('regless', len(regless_clbits))

    unitary_gates = []

    bit_labels = {bit: "%s[%d]" % (reg.name, idx)
                  for reg in qc.qregs + qc.cregs
                  for (idx, bit) in enumerate(reg)}

    bit_labels.update({bit: "regless[%d]" % idx
                       for reg in (regless_qubits, regless_clbits)
                       for idx, bit in enumerate(reg)})

    for instruction, qargs, cargs in qc._data:
        if instruction.name == 'measure':
            qubit = qargs[0]
            clbit = cargs[0]
            string_temp += "%s %s -> %s;\n" % (instruction.qasm(),
                                               bit_labels[qubit],
                                               bit_labels[clbit])

        # If instruction is a root gate or a root instruction (in that case, compositive)

        elif (type(instruction) in  # pylint: disable=unidiomatic-typecheck
              [Gate, Instruction] or
              (isinstance(instruction, ControlledGate) and instruction._open_ctrl)):
            if instruction not in existing_composite_circuits:
                if instruction.name in existing_gate_names:
                    old_name = instruction.name
                    instruction.name += "_" + str(id(instruction))

                    warnings.warn("A gate named {} already exists. "
                                  "We have renamed "
                                  "your gate to {}".format(old_name, instruction.name))

                # Get qasm of composite circuit
                qasm_string = qc._get_composite_circuit_qasm_from_instruction(instruction)

                # Insert composite circuit qasm definition right after header and extension lib
                string_temp = string_temp.replace(qc.extension_lib,
                                                  "%s\n%s" % (qc.extension_lib,
                                                              qasm_string))

                existing_composite_circuits.append(instruction)
                existing_gate_names.append(instruction.name)

            # Insert qasm representation of the original instruction
            string_temp += "%s %s;\n" % (instruction.qasm(),
                                         ",".join([bit_labels[j]
                                                   for j in qargs + cargs]))
        else:
            string_temp += "%s %s;\n" % (instruction.qasm(),
                                         ",".join([bit_labels[j]
                                                   for j in qargs + cargs]))
        if instruction.name == 'unitary':
            unitary_gates.append(instruction)

    # this resets them, so if another call to qasm() is made the gate def is added again
    for gate in unitary_gates:
        gate._qasm_def_written = False

    return string_temp
