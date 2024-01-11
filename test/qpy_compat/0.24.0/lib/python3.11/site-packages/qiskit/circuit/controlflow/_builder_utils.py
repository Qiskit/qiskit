# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Private utility functions that are used by the builder interfaces."""

from typing import Iterable, Tuple, Set

from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.register import Register
from qiskit.circuit.classicalregister import ClassicalRegister
from qiskit.circuit.quantumregister import QuantumRegister


def partition_registers(
    registers: Iterable[Register],
) -> Tuple[Set[QuantumRegister], Set[ClassicalRegister]]:
    """Partition a sequence of registers into its quantum and classical registers."""
    qregs = set()
    cregs = set()
    for register in registers:
        if isinstance(register, QuantumRegister):
            qregs.add(register)
        elif isinstance(register, ClassicalRegister):
            cregs.add(register)
        else:
            # Purely defensive against Terra expansion.
            raise CircuitError(f"Unknown register: {register}.")
    return qregs, cregs


def unify_circuit_resources(circuits: Iterable[QuantumCircuit]) -> Iterable[QuantumCircuit]:
    """
    Ensure that all the given ``circuits`` have all the same qubits, clbits and registers, and
    that they are defined in the same order.  The order is important for binding when the bodies are
    used in the 3-tuple :obj:`.Instruction` context.

    This function will preferentially try to mutate its inputs if they share an ordering, but if
    not, it will rebuild two new circuits.  This is to avoid coupling too tightly to the inner
    class; there is no real support for deleting or re-ordering bits within a :obj:`.QuantumCircuit`
    context, and we don't want to rely on the *current* behaviour of the private APIs, since they
    are very liable to change.  No matter the method used, circuits with unified bits and registers
    are returned.
    """
    circuits = tuple(circuits)
    if len(circuits) < 2:
        return circuits
    qubits = []
    clbits = []
    for circuit in circuits:
        if circuit.qubits[: len(qubits)] != qubits:
            return _unify_circuit_resources_rebuild(circuits)
        if circuit.clbits[: len(qubits)] != clbits:
            return _unify_circuit_resources_rebuild(circuits)
        if circuit.num_qubits > len(qubits):
            qubits = list(circuit.qubits)
        if circuit.num_clbits > len(clbits):
            clbits = list(circuit.clbits)
    for circuit in circuits:
        circuit.add_bits(qubits[circuit.num_qubits :])
        circuit.add_bits(clbits[circuit.num_clbits :])
    return _unify_circuit_registers(circuits)


def _unify_circuit_resources_rebuild(  # pylint: disable=invalid-name  # (it's too long?!)
    circuits: Tuple[QuantumCircuit, ...]
) -> Tuple[QuantumCircuit, QuantumCircuit]:
    """
    Ensure that all the given circuits have all the same qubits and clbits, and that they
    are defined in the same order.  The order is important for binding when the bodies are used in
    the 3-tuple :obj:`.Instruction` context.

    This function will always rebuild the objects into new :class:`.QuantumCircuit` instances.
    """
    qubits, clbits = set(), set()
    for circuit in circuits:
        qubits.update(circuit.qubits)
        clbits.update(circuit.clbits)
    qubits, clbits = list(qubits), list(clbits)

    # We use the inner `_append` method because everything is already resolved in the builders.
    out_circuits = []
    for circuit in circuits:
        out = QuantumCircuit(qubits, clbits, *circuit.qregs, *circuit.cregs)
        for instruction in circuit.data:
            out._append(instruction)
        out_circuits.append(out)
    return _unify_circuit_registers(out_circuits)


def _unify_circuit_registers(circuits: Iterable[QuantumCircuit]) -> Iterable[QuantumCircuit]:
    """
    Ensure that ``true_body`` and ``false_body`` have the same registers defined within them.  These
    do not need to be in the same order between circuits.  The two input circuits are returned,
    mutated to have the same registers.
    """
    circuits = tuple(circuits)
    total_registers = set()
    for circuit in circuits:
        total_registers.update(circuit.qregs)
        total_registers.update(circuit.cregs)
    for circuit in circuits:
        for register in total_registers - set(circuit.qregs) - set(circuit.cregs):
            circuit.add_register(register)
    return circuits
