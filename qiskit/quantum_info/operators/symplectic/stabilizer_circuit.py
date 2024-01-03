# This code is part of Qiskit.
#
# (C) Copyright IBM 2017--2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Stabilizer to circuit function
"""
from __future__ import annotations

from qiskit.exceptions import QiskitError

from qiskit.circuit import QuantumCircuit

from .clifford import Clifford
from .pauli import Pauli


def add_sign(stabilizer: str) -> str:
    """
    Add a sign to stabilizer if it is missing.
    :param stabilizer: stabilizer string
    :return: stabilizer string with sign
    """
    if stabilizer[0] not in ["+", "-"]:
        return "+" + stabilizer
    return stabilizer


def drop_sign(stabilizer: str) -> str:
    """
    Drop sign from stabilizer if it is present.
    :param stabilizer: stabilizer string
    :return: stabilizer string without sign
    """
    if stabilizer[0] not in ["+", "-"]:
        return stabilizer
    if stabilizer[1] == "i":
        return stabilizer[2:]
    return stabilizer[1:]


def check_stabilizers_commutator(s_1: str, s_2: str) -> bool:
    """
    Check if two stabilizers commute.
    :param s_1: stabilizer string
    :param s_2: stabilizer string
    :return: True if stabilizers commute, False otherwise
    """
    prod = 1
    for o1, o2 in zip(drop_sign(s_1), drop_sign(s_2)):
        if o1 == "I" or o2 == "I":
            continue
        if o1 != o2:
            prod *= -1
    return prod == 1


def apply_circuit_on_stabilizer(stabilizer: str, circuit: QuantumCircuit) -> str:
    """
    Given a stabilizer string and a circuit, conjugate the circuit on the stabilizer.
    :param stabilizer: stabilizer string
    :param circuit: Clifford circuit to apply
    :return: stabilizer string after conjugation
    """
    cliff = Clifford(circuit)
    stab_operator = Pauli(stabilizer)
    pauli_conjugated = stab_operator.evolve(cliff, frame="s")
    return pauli_conjugated.to_label()


def stabilizer_to_circuit(
    stabilizer_list: list[str],
    allow_redundant: bool = False,
    allow_underconstrained: bool = False,
    invert: bool = False,
) -> QuantumCircuit:
    """
    Convert a list of stabilizers to a circuit that generates state stabilized by that list using
    Gaussian elimination with Clifford gates.
    Based on stim implementation [1,2]

    [1] https://github.com/quantumlib/Stim/blob/c0dd0b1c8125b2096cd54b6f72884a459e47fe3e/src/stim/stabilizers/conversions.inl#L469 # pylint: disable=line-too-long
    [2] https://quantumcomputing.stackexchange.com/questions/12721/how-to-calculate-destabilizer-group-of-toric-and-other-codes # pylint: disable=line-too-long

    :param stabilizer_list: list of stabilizer strings
    :param allow_redundant: allow redundant stabilizers
    :param allow_underconstrained: allow underconstrained set of stabilizers
    :param invert: return inverse circuit
    :return: QuantumCircuit that generates state stabilized by stabilizer_list
    """

    for i in range(len(stabilizer_list)):
        for j in range(i + 1, len(stabilizer_list)):
            if not check_stabilizers_commutator(stabilizer_list[i], stabilizer_list[j]):
                raise QiskitError(
                    f"Stabilizers {i} ({stabilizer_list[i]}) and {j} ({stabilizer_list[j]}) "
                    "do not commute"
                )
    num_qubits = len(add_sign(stabilizer_list[0])) - 1
    circuit = QuantumCircuit(num_qubits)

    used = 0
    for i in range(len(stabilizer_list)):
        curr_stab = add_sign(apply_circuit_on_stabilizer(stabilizer_list[i], circuit))

        # Find pivot.
        pivot = used + 1
        while pivot <= num_qubits:
            if curr_stab[pivot] != "I":
                break
            pivot += 1
        pivot_index = num_qubits - pivot

        if pivot == num_qubits + 1:
            if "X" in curr_stab or "Y" in curr_stab:
                raise QiskitError(
                    f"Stabilizer {i} ({stabilizer_list[i]}) anti-commutes with some of "
                    "the previous stabilizers"
                )
            if curr_stab[0] == "-":
                raise QiskitError(
                    f"Stabilizer {i} ({stabilizer_list[i]}) contradicts "
                    "some of the previous stabilizers"
                )
            if "Z" in curr_stab and not allow_redundant:
                raise QiskitError(
                    f"Stabilizer {i} ({stabilizer_list[i]}) is a product of the others "
                    "and allow_redundant is False. Add allow_redundant=True "
                    "to the function call if you want to allow redundant stabilizers."
                )
            continue

        # Change pivot basis to the Z axis.
        if curr_stab[pivot] == "X":
            circuit.h(pivot_index)
        elif curr_stab[pivot] == "Y":
            circuit.h(pivot_index)
            circuit.s(pivot_index)
            circuit.h(pivot_index)
            circuit.s(pivot_index)
            circuit.s(pivot_index)

        # Cancel other terms in Pauli string.
        for j in range(1, num_qubits + 1):
            j_index = num_qubits - j
            if j_index == pivot_index or curr_stab[j] == "I":
                continue
            if curr_stab[j] == "X":
                circuit.h(pivot_index)
                circuit.cx(pivot_index, j_index)
                circuit.h(pivot_index)
            elif curr_stab[j] == "Y":
                circuit.h(pivot_index)
                circuit.s(j_index)
                circuit.s(j_index)
                circuit.s(j_index)
                circuit.cx(pivot_index, j_index)
                circuit.h(pivot_index)
                circuit.s(j_index)
            elif curr_stab[j] == "Z":
                circuit.cx(j_index, pivot_index)

        # Move pivot to diagonal.
        used_index = num_qubits - used - 1
        if pivot_index != used_index:
            circuit.swap(pivot_index, used_index)

        # fix sign
        curr_stab = add_sign(apply_circuit_on_stabilizer(stabilizer_list[i], circuit))
        if curr_stab[0] == "-":
            circuit.x(used_index)
        used += 1

    if used < num_qubits and not allow_underconstrained:
        raise QiskitError(
            "Stabilizers are underconstrained and allow_underconstrained is False."
            " Add allow_underconstrained=True  to the function call "
            "if you want to allow underconstrained stabilizers."
        )
    if invert:
        return circuit
    return circuit.inverse()
