# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Helper function for converting a circuit to an instruction"""

from qiskit.exceptions import QiskitError
from qiskit.circuit import Gate
from qiskit.circuit import Reset


def circuit_to_gate(circuit):
    """Build a ``Gate`` object from a ``QuantumCircuit``.

    The gate is anonymous (not tied to a named quantum register),
    and so can be inserted into another circuit. The gate will
    have the same string name as the circuit.

    Args:
        circuit (QuantumCircuit): the input circuit.

    Return:
        Gate: a gate equivalent to the action of the input circuit.
              Upon decomposition, this gate will
              yield the components comprising the original circuit.

    Raises:
        QiskitError: if the circuit is non-reversible and thus not a Gate.
    """
    insts = [inst_context[0] for inst_context in circuit.data]
    if circuit.cregs or any([isinstance(x, Reset) for x in insts]):
        raise QiskitError("circuit %s is non-reversible, and cannot be "
                          "converted to Gate." % circuit.name)

    instruction = Gate(name=circuit.name,
                       num_qubits=sum([qreg.size for qreg in circuit.qregs]),
                       params=[])
    instruction.control = None
    instruction.circuit = None

    instruction.definition = circuit.data

    return instruction
