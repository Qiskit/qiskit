# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Stabilizer to circuit function
"""
from __future__ import annotations

from collections.abc import Collection

import numpy as np

from qiskit.quantum_info import PauliList
from qiskit.exceptions import QiskitError
from qiskit.circuit import QuantumCircuit

from qiskit._accelerate.synthesis.stabilizer import (
    synth_circuit_from_stabilizers as synth_circuit_from_stabilizers_inner,
)


def synth_circuit_from_stabilizers(
    stabilizers: Collection[str],
    allow_redundant: bool = False,
    allow_underconstrained: bool = False,
    invert: bool = False,
) -> QuantumCircuit:
    """Synthesis of a circuit that generates a state stabilized by the stabilizers
    using Gaussian elimination with Clifford gates.
    If the stabilizers are underconstrained, and ``allow_underconstrained`` is ``True``,
    the circuit will output one of the states stabilized by the stabilizers.
    Based on stim implementation.

    Args:
        stabilizers: List of stabilizer strings
        allow_redundant: Allow redundant stabilizers (i.e., some stabilizers
            can be products of the others)
        allow_underconstrained: Allow underconstrained set of stabilizers (i.e.,
            the stabilizers do not specify a unique state)
        invert: Return inverse circuit

    Returns:
        A circuit that generates a state stabilized by ``stabilizers``.

    Raises:
        QiskitError: if the stabilizers are invalid, do not commute, or contradict each other,
                     if the list is underconstrained and ``allow_underconstrained`` is ``False``,
                     or if the list is redundant and ``allow_redundant`` is ``False``.

    References:
        1. https://github.com/quantumlib/Stim/blob/c0dd0b1c8125b2096cd54b6f72884a459e47fe3e/src/stim/stabilizers/conversions.inl#L469
        2. https://quantumcomputing.stackexchange.com/questions/12721/how-to-calculate-destabilizer-group-of-toric-and-other-codes

    """
    stabilizer_list = PauliList(stabilizers)
    if np.any(stabilizer_list.phase % 2):
        raise QiskitError("Some stabilizers have an invalid phase")
    if len(stabilizer_list.commutes_with_all(stabilizer_list)) < len(stabilizer_list):
        raise QiskitError("Some stabilizers do not commute.")

    return QuantumCircuit._from_circuit_data(
        synth_circuit_from_stabilizers_inner(
            stabilizer_list.z.astype(bool),
            stabilizer_list.x.astype(bool),
            stabilizer_list.phase.astype(np.uint8),
            [str(stabilizer) for stabilizer in stabilizer_list],
            allow_redundant,
            allow_underconstrained,
            invert,
        ),
        legacy_qubits=True,
    )
