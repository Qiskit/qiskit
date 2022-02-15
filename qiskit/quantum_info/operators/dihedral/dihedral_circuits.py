# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Circuit simulation for the CNOTDihedral class
"""

import numpy as np

from qiskit.exceptions import QiskitError
from qiskit.circuit import QuantumCircuit


def _append_circuit(elem, circuit, qargs=None):
    """Update a CNOTDihedral element inplace by applying a CNOTDihedral circuit.

    Args:
        elem (CNOTDihedral): the CNOTDihedral element to update.
        circuit (QuantumCircuit or Instruction): the gate or composite gate to apply.
        qargs (list or None): The qubits to apply gates to.
    Returns:
        CNOTDihedral: the updated CNOTDihedral.
    Raises:
        QiskitError: if input gates cannot be decomposed into CNOTDihedral gates.
    """

    if qargs is None:
        qargs = list(range(elem.num_qubits))

    if isinstance(circuit, QuantumCircuit):
        gate = circuit.to_instruction()
    else:
        gate = circuit

    # Handle cx, cz and id since they are basic gates, and cannot be decomposed,
    if gate.name == "cx":
        if len(qargs) != 2:
            raise QiskitError("Invalid qubits for 2-qubit gate cx.")
        elem._append_cx(qargs[0], qargs[1])
        return elem

    elif gate.name == "cz":
        if len(qargs) != 2:
            raise QiskitError("Invalid qubits for 2-qubit gate cz.")
        elem._append_phase(7, qargs[1])
        elem._append_phase(7, qargs[0])
        elem._append_cx(qargs[1], qargs[0])
        elem._append_phase(2, qargs[0])
        elem._append_cx(qargs[1], qargs[0])
        elem._append_phase(7, qargs[1])
        elem._append_phase(7, qargs[0])
        return elem

    if gate.name == "id":
        if len(qargs) != 1:
            raise QiskitError("Invalid qubits for 1-qubit gate id.")
        return elem

    if gate.definition is None:
        raise QiskitError(f"Cannot apply Instruction: {gate.name}")
    if not isinstance(gate.definition, QuantumCircuit):
        raise QiskitError(
            "{} instruction definition is {}; expected QuantumCircuit".format(
                gate.name, type(gate.definition)
            )
        )

    flat_instr = gate.definition
    bit_indices = {
        bit: index
        for bits in [flat_instr.qubits, flat_instr.clbits]
        for index, bit in enumerate(bits)
    }

    for instr, qregs, _ in gate.definition:
        # Get the integer position of the flat register
        new_qubits = [qargs[bit_indices[tup]] for tup in qregs]

        if instr.name == "x" or gate.name == "x":
            if len(new_qubits) != 1:
                raise QiskitError("Invalid qubits for 1-qubit gate x.")
            elem._append_x(new_qubits[0])

        elif instr.name == "z" or gate.name == "z":
            if len(new_qubits) != 1:
                raise QiskitError("Invalid qubits for 1-qubit gate z.")
            elem._append_phase(4, new_qubits[0])

        elif instr.name == "y" or gate.name == "y":
            if len(new_qubits) != 1:
                raise QiskitError("Invalid qubits for 1-qubit gate y.")
            elem._append_x(new_qubits[0])
            elem._append_phase(4, new_qubits[0])

        elif instr.name == "p" or gate.name == "p":
            if len(new_qubits) != 1 or len(instr.params) != 1:
                raise QiskitError("Invalid qubits or params for 1-qubit gate p.")
            elem._append_phase(int(4 * instr.params[0] / np.pi), new_qubits[0])

        elif instr.name == "t" or gate.name == "t":
            if len(new_qubits) != 1:
                raise QiskitError("Invalid qubits for 1-qubit gate t.")
            elem._append_phase(1, new_qubits[0])

        elif instr.name == "tdg" or gate.name == "tdg":
            if len(new_qubits) != 1:
                raise QiskitError("Invalid qubits for 1-qubit gate tdg.")
            elem._append_phase(7, new_qubits[0])

        elif instr.name == "s" or gate.name == "s":
            if len(new_qubits) != 1:
                raise QiskitError("Invalid qubits for 1-qubit gate s.")
            elem._append_phase(2, new_qubits[0])

        elif instr.name == "sdg" or gate.name == "sdg":
            if len(new_qubits) != 1:
                raise QiskitError("Invalid qubits for 1-qubit gate sdg.")
            elem._append_phase(6, new_qubits[0])

        elif instr.name == "cx":
            if len(new_qubits) != 2:
                raise QiskitError("Invalid qubits for 2-qubit gate cx.")
            elem._append_cx(new_qubits[0], new_qubits[1])

        elif instr.name == "cz":
            if len(new_qubits) != 2:
                raise QiskitError("Invalid qubits for 2-qubit gate cz.")
            elem._append_phase(7, new_qubits[1])
            elem._append_phase(7, new_qubits[0])
            elem._append_cx(new_qubits[1], new_qubits[0])
            elem._append_phase(2, new_qubits[0])
            elem._append_cx(new_qubits[1], new_qubits[0])
            elem._append_phase(7, new_qubits[1])
            elem._append_phase(7, new_qubits[0])

        elif instr.name == "swap" or gate.name == "swap":
            if len(new_qubits) != 2:
                raise QiskitError("Invalid qubits for 2-qubit gate swap.")
            elem._append_cx(new_qubits[0], new_qubits[1])
            elem._append_cx(new_qubits[1], new_qubits[0])
            elem._append_cx(new_qubits[0], new_qubits[1])

        elif instr.name == "id":
            pass

        else:
            raise QiskitError(f"Not a CNOT-Dihedral gate: {instr.name}")

    return elem
