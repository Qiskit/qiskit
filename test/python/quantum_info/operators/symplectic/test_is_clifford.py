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

"""Tests for is_clifford utility function."""

from __future__ import annotations

import numpy as np
import pytest

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, Barrier, Delay
from qiskit.circuit.library import (
    HGate,
    XGate,
    YGate,
    ZGate,
    SGate,
    SdgGate,
    SXGate,
    SXdgGate,
    CXGate,
    CZGate,
    CYGate,
    SwapGate,
    iSwapGate,
    ECRGate,
    DCXGate,
    TGate,
    TdgGate,
    CCXGate,
    CCZGate,
    RXGate,
    RYGate,
    RZGate,
    UGate,
    LinearFunction,
    PermutationGate,
)
from qiskit.quantum_info import Clifford, Pauli, PauliList, Operator, StabilizerState
from qiskit.quantum_info import is_clifford


class TestIsClifford:
    """Test suite for is_clifford utility function."""

    def test_canonical_clifford_objects(self):
        """Test is_clifford on Clifford and Pauli objects."""
        c = Clifford.from_label("XYZ")
        assert is_clifford(c) is True

        p = Pauli("XYZ")
        assert is_clifford(p) is True

        pl = PauliList(["XYZ", "III"])
        assert is_clifford(pl) is True

        state = StabilizerState(Clifford.from_label("X"))
        assert is_clifford(state) is True

    def test_standard_clifford_gates(self):
        """Test is_clifford on individual standard 1Q and 2Q Clifford gates."""
        clifford_gates = [
            HGate(),
            XGate(),
            YGate(),
            ZGate(),
            SGate(),
            SdgGate(),
            SXGate(),
            SXdgGate(),
            CXGate(),
            CZGate(),
            CYGate(),
            SwapGate(),
            iSwapGate(),
            ECRGate(),
            DCXGate(),
        ]
        for gate in clifford_gates:
            assert is_clifford(gate) is True, f"Failed for {gate.name}"

    def test_standard_non_clifford_gates(self):
        """Test is_clifford on standard non-Clifford gates."""
        non_clifford_gates = [
            TGate(),
            TdgGate(),
            CCXGate(),
            CCZGate(),
            RXGate(0.1),
            RYGate(np.pi / 3),
            RZGate(0.7),
            UGate(0.1, 0.2, 0.3),
        ]
        for gate in non_clifford_gates:
            assert is_clifford(gate) is False, f"Should be False for {gate.name}"

    def test_rotation_gates_at_multiples_of_pi_over_two(self):
        """Test rotation gates with angles that are multiples of pi/2."""
        assert is_clifford(RXGate(np.pi / 2)) is True
        assert is_clifford(RXGate(np.pi)) is True
        assert is_clifford(RYGate(np.pi / 2)) is True
        assert is_clifford(RZGate(3 * np.pi / 2)) is True
        assert is_clifford(UGate(np.pi / 2, 0, np.pi)) is True

        # Non-multiples
        assert is_clifford(RXGate(np.pi / 4)) is False
        assert is_clifford(RZGate(np.pi / 6)) is False

    def test_special_library_gates(self):
        """Test LinearFunction, PermutationGate, Barrier, Delay."""
        lin = LinearFunction([[1, 1], [0, 1]])
        assert is_clifford(lin) is True

        perm = PermutationGate([1, 0, 2])
        assert is_clifford(perm) is True

        assert is_clifford(Barrier(2)) is True
        assert is_clifford(Delay(100, unit="ns")) is True

    def test_pure_clifford_circuits(self):
        """Test circuits composed solely of Clifford gates."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cz(1, 2)
        qc.s(2)
        qc.swap(0, 2)
        assert is_clifford(qc) is True

    def test_non_clifford_circuits(self):
        """Test circuits containing non-Clifford gates."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.t(0)
        qc.cx(0, 1)
        assert is_clifford(qc) is False

    def test_cancelling_non_clifford_circuits(self):
        """Test circuits where intermediate non-Clifford gates cancel out."""
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.t(0)
        qc.tdg(0)
        # H followed by T and Tdg is equivalent to H, which is Clifford!
        assert is_clifford(qc) is True

        qc2 = QuantumCircuit(2)
        qc2.rx(0.3, 0)
        qc2.rx(-0.3, 0)
        qc2.cx(0, 1)
        assert is_clifford(qc2) is True

    def test_circuit_with_classical_bits(self):
        """Test that circuits with classical bits / measurements return False."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.measure(0, 0)
        assert is_clifford(qc) is False

    def test_parameterized_circuits(self):
        """Test that parameterized circuits with unbound parameters return False."""
        theta = Parameter("theta")
        qc = QuantumCircuit(1)
        qc.rx(theta, 0)
        assert is_clifford(qc) is False

        # When bound to a Clifford angle:
        bound_qc = qc.assign_parameters({theta: np.pi / 2})
        assert is_clifford(bound_qc) is True

    def test_quantum_info_operator(self):
        """Test Operator objects."""
        op_h = Operator.from_label("H")
        assert is_clifford(op_h) is True

        op_t = Operator(TGate())
        assert is_clifford(op_t) is False

        # Non-unitary operator
        op_proj = Operator(np.array([[1, 0], [0, 0]]))
        assert is_clifford(op_proj) is False

    def test_numpy_matrix(self):
        """Test raw numpy matrices."""
        h_mat = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        assert is_clifford(h_mat) is True

        t_mat = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
        assert is_clifford(t_mat) is False

        # Non-square or invalid dimension
        assert is_clifford(np.array([1, 0])) is False
        assert is_clifford(np.zeros((3, 3))) is False

    def test_unsupported_types(self):
        """Test unsupported types return False gracefully."""
        assert is_clifford("H") is False
        assert is_clifford(123) is False
        assert is_clifford(None) is False
        assert is_clifford([1, 2, 3]) is False
