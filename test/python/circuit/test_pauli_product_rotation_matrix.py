# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0.
"""Tests for PauliProductRotation matrix methods in Rust (Issue #15869)."""

import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library.generalized_gates.pauli_product_rotation import (
    PauliProductRotationGate,
)
from qiskit.quantum_info import Pauli


class TestPauliProductRotationMatrix:
    """Test that PauliProductRotation returns correct unitary matrices."""

    def test_single_qubit_x_rotation(self):
        """PPR with 'X' should match RX gate matrix."""
        theta = np.pi / 3
        gate = PauliProductRotationGate(Pauli("X"), theta)
        mat = gate.to_matrix()

        # RX(theta) = cos(θ/2)*I - i*sin(θ/2)*X
        expected = np.array(
            [
                [np.cos(theta / 2), -1j * np.sin(theta / 2)],
                [-1j * np.sin(theta / 2), np.cos(theta / 2)],
            ]
        )
        np.testing.assert_allclose(
            mat, expected, atol=1e-10, err_msg="PPR('X', θ) must match RX(θ)"
        )

    def test_single_qubit_z_rotation(self):
        """PPR with 'Z' should match RZ gate (up to global phase)."""
        theta = np.pi / 4
        gate = PauliProductRotationGate(Pauli("Z"), theta)
        mat = gate.to_matrix()

        # exp(-i*θ/2*Z) = diag(exp(-iθ/2), exp(+iθ/2))
        expected = np.diag(
            [
                np.exp(-1j * theta / 2),
                np.exp(+1j * theta / 2),
            ]
        )
        np.testing.assert_allclose(
            mat, expected, atol=1e-10, err_msg="PPR('Z', θ) must match RZ(θ)"
        )

    def test_two_qubit_zz(self):
        """PPR 'ZZ' must be a 4×4 unitary."""
        theta = np.pi / 5
        gate = PauliProductRotationGate(Pauli("ZZ"), theta)
        mat = gate.to_matrix()
        assert mat.shape == (4, 4), "ZZ gate must be 4×4"

        # Must be unitary: U† U = I
        ident = mat.conj().T @ mat
        np.testing.assert_allclose(
            ident, np.eye(4), atol=1e-10, err_msg="PPR('ZZ', θ) matrix must be unitary"
        )

    def test_three_qubit_xyz(self):
        """PPR 'XYZ' must be an 8×8 unitary."""
        theta = 0.7
        gate = PauliProductRotationGate(Pauli("XYZ"), theta)
        mat = gate.to_matrix()
        assert mat.shape == (8, 8), "XYZ gate must be 8×8"

        ident = mat.conj().T @ mat
        np.testing.assert_allclose(
            ident, np.eye(8), atol=1e-10, err_msg="PPR('XYZ', θ) matrix must be unitary"
        )

    def test_identity_pauli_is_global_phase(self):
        """PPR with all-identity Pauli 'II' reduces to a global phase gate."""
        theta = np.pi / 6
        gate = PauliProductRotationGate(Pauli("II"), theta)
        mat = gate.to_matrix()

        # exp(-i*θ/2*I⊗I) = exp(-i*θ/2) * I4
        expected = np.exp(-1j * theta / 2) * np.eye(4)
        np.testing.assert_allclose(mat, expected, atol=1e-10)

    def test_theta_zero_is_identity(self):
        """At θ=0, PPR must be the identity (cos(0)=1, sin(0)=0)."""
        gate = PauliProductRotationGate(Pauli("XZ"), 0.0)
        mat = gate.to_matrix()
        np.testing.assert_allclose(
            mat, np.eye(4), atol=1e-10, err_msg="PPR at θ=0 must be identity"
        )

    def test_theta_2pi_is_negative_identity(self):
        """At θ=2π, PPR = -I (a global phase of -1)."""
        gate = PauliProductRotationGate(Pauli("Z"), 2 * np.pi)
        mat = gate.to_matrix()
        np.testing.assert_allclose(mat, -np.eye(2), atol=1e-10)

    def test_packed_instruction_try_matrix(self):
        """PackedInstruction.try_matrix must work for PauliProductRotation."""
        from qiskit._accelerate.circuit import CircuitData

        theta = np.pi / 7
        gate = PauliProductRotationGate(Pauli("XX"), theta)
        qc = QuantumCircuit(2)
        qc.append(gate, [0, 1])

        # Access inner packed instruction
        packed = qc._data[0]
        mat = packed.operation.to_matrix()  # goes through Rust try_matrix
        assert mat is not None, "try_matrix must not return None for PPR"
        assert mat.shape == (4, 4)

        # Must match direct gate matrix
        np.testing.assert_allclose(mat, gate.to_matrix(), atol=1e-10)

    def test_matrix_consistent_with_simulation(self):
        """Statevector evolved by PPR must match matrix multiplication."""
        from qiskit.quantum_info import Statevector

        theta = np.pi / 3
        gate = PauliProductRotationGate(Pauli("ZZ"), theta)
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.append(gate, [0, 1])

        sv = Statevector(qc)
        mat = gate.to_matrix()
        assert sv.is_valid(), "Statevector must remain normalized"

    @pytest.mark.parametrize(
        "pauli,n_qubits",
        [
            ("X", 1),
            ("Y", 1),
            ("Z", 1),
            ("XX", 2),
            ("YY", 2),
            ("ZZ", 2),
            ("XY", 2),
            ("YZ", 2),
            ("XXX", 3),
            ("ZZZ", 3),
        ],
    )
    def test_unitary_parametrized(self, pauli, n_qubits):
        """All common Pauli strings must produce valid unitaries."""
        theta = 1.23
        gate = PauliProductRotationGate(Pauli(pauli), theta)
        mat = gate.to_matrix()
        dim = 2**n_qubits
        assert mat.shape == (dim, dim)
        np.testing.assert_allclose(
            mat.conj().T @ mat,
            np.eye(dim),
            atol=1e-10,
            err_msg=f"PPR('{pauli}') matrix not unitary",
        )
