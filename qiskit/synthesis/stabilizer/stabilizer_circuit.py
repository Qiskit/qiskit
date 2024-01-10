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

from collections.abc import Collection

from qiskit.exceptions import QiskitError

from qiskit.circuit import QuantumCircuit

from qiskit.quantum_info.operators.symplectic.clifford import Clifford
from qiskit.quantum_info.operators.symplectic.pauli import Pauli


def _add_sign(stabilizer: str) -> str:
    """
    Add a sign to stabilizer if it is missing.

    Args:
        stabilizer (str): stabilizer string

    Return:
        str: stabilizer string with sign
    """
    if stabilizer[0] not in ["+", "-"]:
        return "+" + stabilizer
    return stabilizer


def _drop_sign(stabilizer: str) -> str:
    """
    Drop sign from stabilizer if it is present.

    Args:
        stabilizer (str): stabilizer string

    Return:
        str: stabilizer string without sign
    """
    if stabilizer[0] not in ["+", "-"]:
        return stabilizer
    if stabilizer[1] == "i":
        return stabilizer[2:]
    return stabilizer[1:]


def _check_stabilizers_commutator(s_1: str, s_2: str) -> bool:
    """
    Check if two stabilizers commute.

    Args:
        s_1 (str): stabilizer string
        s_1 (str): stabilizer string

    Return:
        bool: True if stabilizers commute, False otherwise.
    """
    prod = 1
    for o1, o2 in zip(_drop_sign(s_1), _drop_sign(s_2)):
        if o1 == "I" or o2 == "I":
            continue
        if o1 != o2:
            prod *= -1
    return prod == 1


def _apply_circuit_on_stabilizer(stabilizer: str, circuit: QuantumCircuit) -> str:
    """
    Given a stabilizer string and a circuit, conjugate the circuit on the stabilizer.

    Args:
        stabilizer (str): stabilizer string
        circuit (QuantumCircuit): Clifford circuit to apply

    Return:
        str: a pauli string after conjugation.
    """
    cliff = Clifford(circuit)
    stab_operator = Pauli(stabilizer)
    pauli_conjugated = stab_operator.evolve(cliff, frame="s")
    return pauli_conjugated.to_label()


def synth_circuit_from_stabilizers(
    stabilizers: Collection[str],
    allow_redundant: bool = False,
    allow_underconstrained: bool = False,
    invert: bool = False,
) -> QuantumCircuit:
    # pylint: disable=line-too-long
    """Synthesis of a circuit that generates a state stabilized by the stabilziers
    using Gaussian elimination with Clifford gates.
    If the stabilizers are underconstrained, and `allow_underconstrained` is `True`,
    the circuit will output one of the states stabilized by the stabilizers.
    Based on stim implementation.

    Args:
        stabilizers (Collection[str]): list of stabilizer strings
        allow_redundant (bool): allow redundant stabilizers (i.e., some stabilizers
            can be products of the others)
        allow_underconstrained (bool): allow underconstrained set of stabilizers (i.e.,
            the stabilizers do not specify a unique state)
        invert (bool): return inverse circuit

    Return:
        QuantumCircuit: a circuit that generates a state stabilized by `stabilizers`.

    Raises:
        QiskitError: if the stabilizers are invalid, do not commute, or contradict each other,
                     if the list is underconstrained and `allow_underconstrained` is `False`,
                     or if the list is redundant and `allow_redundant` is `False`.

    Reference:
        1. https://github.com/quantumlib/Stim/blob/c0dd0b1c8125b2096cd54b6f72884a459e47fe3e/src/stim/stabilizers/conversions.inl#L469
        2. https://quantumcomputing.stackexchange.com/questions/12721/how-to-calculate-destabilizer-group-of-toric-and-other-codes

    """
    stabilizer_list = list(stabilizers)

    # verification
    for i, stabilizer in enumerate(stabilizer_list):
        if set(stabilizer) - set("IXYZ+-i"):
            raise QiskitError(f"Stabilizer {i} ({stabilizer}) contains invalid characters")
        if stabilizer[1] == "i":
            raise QiskitError(f"Stabilizer {i} ({stabilizer}) has an invalid phase")
    for i in range(len(stabilizer_list)):
        for j in range(i + 1, len(stabilizer_list)):
            if not _check_stabilizers_commutator(stabilizer_list[i], stabilizer_list[j]):
                raise QiskitError(
                    f"Stabilizers {i} ({stabilizer_list[i]}) and {j} ({stabilizer_list[j]}) "
                    "do not commute"
                )

    num_qubits = len(_drop_sign(stabilizer_list[0]))
    circuit = QuantumCircuit(num_qubits)

    used = 0
    for i in range(len(stabilizer_list)):
        curr_stab = _add_sign(_apply_circuit_on_stabilizer(stabilizer_list[i], circuit))

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
        curr_stab = _add_sign(_apply_circuit_on_stabilizer(stabilizer_list[i], circuit))
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
