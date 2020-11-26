# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
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

    # Handle cx since it is a basic gate, and cannot be decomposed,
    # so gate.definition = None
    if gate.name == 'cx':
        if len(qargs) != 2:
            raise QiskitError("Invalid qubits for 2-qubit gate cx.")
        elem.cnot(qargs[0], qargs[1])
        return elem

    if gate.definition is None:
        raise QiskitError('Cannot apply Instruction: {}'.format(gate.name))
    if not isinstance(gate.definition, QuantumCircuit):
        raise QiskitError('{} instruction definition is {}; expected QuantumCircuit'.format(
            gate.name, type(gate.definition)))

    for instr, qregs, _ in gate.definition:
        # Get the integer position of the flat register
        new_qubits = [qargs[tup.index] for tup in qregs]

        if (instr.name == 'x' or gate.name == 'x'):
            if len(new_qubits) != 1:
                raise QiskitError("Invalid qubits for 1-qubit gate x.")
            elem.flip(new_qubits[0])

        elif (instr.name == 'z' or gate.name == 'z'):
            if len(new_qubits) != 1:
                raise QiskitError("Invalid qubits for 1-qubit gate z.")
            elem.phase(4, new_qubits[0])

        elif (instr.name == 'y' or gate.name == 'y'):
            if len(new_qubits) != 1:
                raise QiskitError("Invalid qubits for 1-qubit gate y.")
            elem.flip(new_qubits[0])
            elem.phase(4, new_qubits[0])

        elif (instr.name == 'p' or gate.name == 'p'):
            if (len(new_qubits) != 1 or len(instr.params) != 1):
                raise QiskitError("Invalid qubits or params for 1-qubit gate p.")
            elem.phase(int(4 * instr.params[0] / np.pi), new_qubits[0])

        elif (instr.name == 'cx' or gate.name == 'cx'):
            if len(new_qubits) != 2:
                raise QiskitError("Invalid qubits for 2-qubit gate cx.")
            elem.cnot(new_qubits[0], new_qubits[1])

        elif (instr.name == 'cz' or gate.name == 'cz'):
            if len(new_qubits) != 2:
                raise QiskitError("Invalid qubits for 2-qubit gate cz.")
            elem.phase(7, new_qubits[1])
            elem.phase(7, new_qubits[0])
            elem.cnot(new_qubits[1], new_qubits[0])
            elem.phase(2, new_qubits[0])
            elem.cnot(new_qubits[1], new_qubits[0])
            elem.phase(7, new_qubits[1])
            elem.phase(7, new_qubits[0])

        elif (instr.name == 'id' or gate.name == 'id'):
            pass

        else:
            raise QiskitError('Not a CNOT-Dihedral gate: {}'.format(gate.name))

    return elem
