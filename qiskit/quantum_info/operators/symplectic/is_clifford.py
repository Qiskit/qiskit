# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Cliffordness checker."""

from __future__ import annotations

from typing import Any

import numpy as np

from qiskit.exceptions import QiskitError
from qiskit.circuit import QuantumCircuit, Instruction, Barrier, Delay
from qiskit.circuit.library import LinearFunction, PermutationGate
from qiskit.quantum_info.operators.symplectic.clifford import Clifford
from qiskit.quantum_info.operators.symplectic.pauli import Pauli
from qiskit.quantum_info.operators.symplectic.pauli_list import PauliList
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.states.stabilizerstate import StabilizerState


def _matrix_is_clifford(matrix: np.ndarray) -> bool:
    """Return ``True`` if *matrix* is a unitary Clifford matrix.

    Uses :meth:`Clifford._unitary_matrix_to_tableau` which implements the
    symplectic check directly: it returns a valid tableau array when the matrix
    is Clifford and ``None`` otherwise.  Non-unitary and non-power-of-2
    matrices are handled safely.

    Args:
        matrix: A square 2-D numpy array.

    Returns:
        bool: ``True`` if *matrix* represents a Clifford operation.
    """
    try:
        return Clifford._unitary_matrix_to_tableau(matrix) is not None
    except Exception:  # noqa: BLE001  (e.g. dimension mismatch)
        return False


def is_clifford(data: Any) -> bool:
    """Check if the input represents a Clifford operation.

    This function tests whether an input object (such as a
    :class:`~qiskit.circuit.QuantumCircuit`,
    :class:`~qiskit.circuit.Instruction` / :class:`~qiskit.circuit.Gate`,
    :class:`~qiskit.quantum_info.Operator`, :class:`~qiskit.quantum_info.Clifford`,
    or a unitary matrix) belongs to the Clifford group.

    A quantum operation is Clifford if and only if it maps every element of
    the Pauli group to an element of the Pauli group under conjugation
    (up to a global phase).

    The check uses a tiered strategy:

    1. **Direct type matching**: Immediately returns ``True`` for inherently
       Clifford types (:class:`~qiskit.quantum_info.Clifford`,
       :class:`~qiskit.quantum_info.Pauli`,
       :class:`~qiskit.quantum_info.PauliList`,
       :class:`~qiskit.quantum_info.StabilizerState`,
       :class:`~qiskit.circuit.library.LinearFunction`,
       :class:`~qiskit.circuit.library.PermutationGate`,
       :class:`~qiskit.circuit.Barrier`, :class:`~qiskit.circuit.Delay`).
    2. **Fast gate-level circuit check**: Evaluates gate sequences via
       :meth:`~qiskit.quantum_info.Clifford.from_circuit` in O(N) gate
       traversal.  Handles rotation gates at angles :math:`k\\pi/2`.
    3. **Exact unitary fallback** (circuits/gates up to 5 qubits,
       ``Operator`` and ``numpy.ndarray`` of any valid qubit dimension):
       Constructs the full unitary matrix and attempts to build a
       :class:`~qiskit.quantum_info.Clifford` from it.  This correctly
       handles circuits where non-Clifford gates cancel each other out
       (e.g. :math:`H \\cdot T \\cdot T^\\dagger \\equiv H`).
    4. **Safety guards**: Returns ``False`` for circuits with classical
       bits / measurements, unassigned parameters, non-unitary matrices,
       or operations that definitively cannot be Clifford.

    Args:
        data: The object to check.  Supported types:

            - :class:`~qiskit.quantum_info.Clifford`
            - :class:`~qiskit.quantum_info.Pauli`
            - :class:`~qiskit.quantum_info.PauliList`
            - :class:`~qiskit.quantum_info.StabilizerState`
            - :class:`~qiskit.circuit.QuantumCircuit`
            - :class:`~qiskit.circuit.Instruction` /
              :class:`~qiskit.circuit.Gate`
            - :class:`~qiskit.quantum_info.Operator`
            - 2-D unitary :class:`numpy.ndarray`

    Returns:
        bool: ``True`` if the input represents a Clifford operation,
        ``False`` otherwise.

    Examples:
        >>> from qiskit import QuantumCircuit
        >>> from qiskit.circuit.library import HGate, TGate
        >>> from qiskit.quantum_info import is_clifford, Clifford, Operator
        >>> is_clifford(HGate())
        True
        >>> is_clifford(TGate())
        False
        >>> qc = QuantumCircuit(1)
        >>> qc.h(0)
        >>> qc.t(0)
        >>> qc.tdg(0)
        >>> is_clifford(qc)
        True
    """
    # ------------------------------------------------------------------ #
    # Tier 1 – canonical Clifford types: always True                      #
    # ------------------------------------------------------------------ #
    if isinstance(data, (Clifford, Pauli, PauliList, StabilizerState)):
        return True

    if isinstance(data, (Barrier, Delay)):
        return True

    if isinstance(data, (LinearFunction, PermutationGate)):
        return True

    # ------------------------------------------------------------------ #
    # Tier 2 – QuantumCircuit and Instruction / Gate                      #
    # ------------------------------------------------------------------ #
    if isinstance(data, (QuantumCircuit, Instruction)):
        # Parameterized circuits with unassigned parameters cannot be
        # evaluated numerically.
        if hasattr(data, "num_parameters") and data.num_parameters > 0:
            return False

        # Classical bits / measurements make the circuit non-unitary.
        if isinstance(data, QuantumCircuit) and data.clbits:
            return False

        # Fast path: attempt direct Clifford circuit construction.
        try:
            Clifford.from_circuit(data)
            return True
        except QiskitError:
            pass

        # Exact fallback: build the full unitary and check Cliffordness.
        # Limited to ≤5 qubits to avoid intractable 2^N×2^N matrices.
        if hasattr(data, "num_qubits") and 0 < data.num_qubits <= 5:
            try:
                return _matrix_is_clifford(Operator(data).data)
            except Exception:  # noqa: BLE001
                return False

        return False

    # ------------------------------------------------------------------ #
    # Tier 3 – Operator                                                   #
    # ------------------------------------------------------------------ #
    if isinstance(data, Operator):
        if not data.is_unitary():
            return False
        return _matrix_is_clifford(data.data)

    # ------------------------------------------------------------------ #
    # Tier 4 – raw numpy matrix                                           #
    # ------------------------------------------------------------------ #
    if isinstance(data, np.ndarray):
        if (
            data.ndim == 2
            and data.shape[0] == data.shape[1]
            and data.shape[0] > 0
            and (data.shape[0] & (data.shape[0] - 1)) == 0  # power-of-2 dimension
        ):
            return _matrix_is_clifford(data)
        return False

    return False
