# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Isometry tests."""

import unittest
import numpy as np
from ddt import ddt, data, unpack

from qiskit.quantum_info import random_unitary
from qiskit import QuantumCircuit
from qiskit import QuantumRegister
from qiskit.compiler import transpile
from qiskit.quantum_info import Operator
from qiskit.circuit.library.generalized_gates import Isometry
from test import QiskitTestCase


@ddt
class TestIsometry(QiskitTestCase):
    """Qiskit isometry tests."""

    @data(
        np.eye(2, 2),
        random_unitary(2, seed=868540).data,
        np.eye(4, 4),
        random_unitary(4, seed=16785).data[:, 0],
        np.eye(4, 4)[:, 0:2],
        random_unitary(4, seed=660477).data,
        np.eye(4, 4)[:, np.random.RandomState(seed=719010).permutation(4)][:, 0:2],
        np.eye(8, 8)[:, np.random.RandomState(seed=544326).permutation(8)],
        random_unitary(8, seed=247924).data[:, 0:4],
        random_unitary(8, seed=765720).data,
        random_unitary(16, seed=278663).data,
        random_unitary(16, seed=406498).data[:, 0:8],
    )
    def test_isometry(self, iso):
        """Tests for the decomposition of isometries from m to n qubits"""
        if len(iso.shape) == 1:
            iso = iso.reshape((len(iso), 1))
        num_q_output = int(np.log2(iso.shape[0]))
        num_q_input = int(np.log2(iso.shape[1]))
        qc = QuantumCircuit(num_q_output)

        gate = Isometry(iso, num_ancillas_zero=0, num_ancillas_dirty=0)
        qc.append(gate, qc.qubits)

        # Verify the circuit can be decomposed
        self.assertIsInstance(qc.decompose(), QuantumCircuit)

        # Decompose the gate
        qc = transpile(qc, basis_gates=["u1", "u3", "u2", "cx", "id"])

        # Simulate the decomposed gate
        unitary = Operator(qc).data
        iso_from_circuit = unitary[::, 0 : 2**num_q_input]
        iso_desired = iso

        self.assertTrue(np.allclose(iso_from_circuit, iso_desired))

    @data(
        np.eye(2, 2),
        random_unitary(2, seed=99506).data,
        np.eye(4, 4),
        random_unitary(4, seed=673459).data[:, 0],
        np.eye(4, 4)[:, 0:2],
        random_unitary(4, seed=124090).data,
        np.eye(4, 4)[:, np.random.RandomState(seed=889848).permutation(4)][:, 0:2],
        np.eye(8, 8)[:, np.random.RandomState(seed=94795).permutation(8)],
        random_unitary(8, seed=986292).data[:, 0:4],
        random_unitary(8, seed=632121).data,
        random_unitary(16, seed=623107).data,
        random_unitary(16, seed=889326).data[:, 0:8],
    )
    def test_isometry_tolerance(self, iso):
        """Tests for the decomposition of isometries from m to n qubits with a custom tolerance"""
        if len(iso.shape) == 1:
            iso = iso.reshape((len(iso), 1))
        num_q_output = int(np.log2(iso.shape[0]))
        num_q_input = int(np.log2(iso.shape[1]))
        qc = QuantumCircuit(num_q_output)

        # Compute isometry with custom tolerance
        gate = Isometry(iso, num_ancillas_zero=0, num_ancillas_dirty=0, epsilon=1e-3)
        qc.append(gate, qc.qubits)

        # Verify the circuit can be decomposed
        self.assertIsInstance(qc.decompose(), QuantumCircuit)

        # Decompose the gate
        qc = transpile(qc, basis_gates=["u1", "u3", "u2", "cx", "id"])

        # Simulate the decomposed gate
        unitary = Operator(qc).data
        iso_from_circuit = unitary[::, 0 : 2**num_q_input]
        self.assertTrue(np.allclose(iso_from_circuit, iso))

    @data(
        np.eye(2, 2),
        random_unitary(2, seed=272225).data,
        np.eye(4, 4),
        random_unitary(4, seed=592640).data[:, 0],
        np.eye(4, 4)[:, 0:2],
        random_unitary(4, seed=714210).data,
        np.eye(4, 4)[:, np.random.RandomState(seed=719934).permutation(4)][:, 0:2],
        np.eye(8, 8)[:, np.random.RandomState(seed=284469).permutation(8)],
        random_unitary(8, seed=656745).data[:, 0:4],
        random_unitary(8, seed=583813).data,
        random_unitary(16, seed=101363).data,
        random_unitary(16, seed=583429).data[:, 0:8],
    )
    def test_isometry_inverse(self, iso):
        """Tests for the inverse of isometries from m to n qubits"""
        if len(iso.shape) == 1:
            iso = iso.reshape((len(iso), 1))

        num_q_output = int(np.log2(iso.shape[0]))

        q = QuantumRegister(num_q_output)
        qc = QuantumCircuit(q)
        qc.append(Isometry(iso, 0, 0), q)
        qc.append(Isometry(iso, 0, 0).inverse(), q)

        result = Operator(qc)
        np.testing.assert_array_almost_equal(result.data, np.identity(result.dim[0]))

    @data(
        np.eye(2, 2),
        random_unitary(2, seed=297102).data,
        np.eye(4, 4),
        random_unitary(4, seed=123642).data,
        random_unitary(8, seed=568288).data,
    )
    def test_isometry_repeat(self, iso):
        """Tests for the repeat of isometries from n to n qubits"""
        iso_gate = Isometry(iso, 0, 0)

        op = Operator(iso_gate)
        op_double = Operator(iso_gate.repeat(2))
        np.testing.assert_array_almost_equal(op @ op, op_double)

    @data(
        (np.eye(2, 2), 1, 0),
        (np.eye(2, 2), 0, 1),
        (random_unitary(4, seed=55021).data[:, 0:2], 1, 0),
        (random_unitary(4, seed=55022).data[:, 0:2], 0, 1),
        (random_unitary(4, seed=55023).data[:, 0:2], 1, 1),
    )
    @unpack
    def test_isometry_with_ancillas(self, iso, num_ancillas_zero, num_ancillas_dirty):
        """The decomposition must stay correct once ancilla qubits are actually used.

        Every other case in this file passes ``num_ancillas_zero=num_ancillas_dirty=0``, so the
        ancilla-handling branches of ``Isometry`` (and of the underlying Rust
        ``synth_isometry``, which widens ``num_qubits`` but currently never *uses* ancillas --
        see the "ancillas are idle" invariant tested on the Rust side) were previously only
        reachable from users' own circuits, not from the test suite. Here the ancillas start in
        the all-zero computational basis state together with the isometry's own input qubits,
        so the standard "read off the first 2**m columns" check from ``test_isometry`` still
        applies directly: it tells us the gate decomposes correctly in the presence of unused
        ancilla qubits, exactly as `qiskit/circuit/library/generalized_gates/isometry.py`'s
        own `inv_gate`/`_define` wires them in.
        """
        num_q_output = int(np.log2(iso.shape[0]))
        num_q_input = int(np.log2(iso.shape[1]))
        total_qubits = num_q_output + num_ancillas_zero + num_ancillas_dirty
        qc = QuantumCircuit(total_qubits)

        gate = Isometry(
            iso, num_ancillas_zero=num_ancillas_zero, num_ancillas_dirty=num_ancillas_dirty
        )
        self.assertEqual(gate.num_qubits, total_qubits)
        qc.append(gate, qc.qubits)

        qc = transpile(qc, basis_gates=["u1", "u3", "u2", "cx", "id"])
        unitary = Operator(qc).data
        iso_from_circuit = unitary[0 : 2**num_q_output, 0 : 2**num_q_input]

        self.assertTrue(np.allclose(iso_from_circuit, iso, atol=1e-7))

    def test_synth_isometry_rejects_non_power_of_two_columns(self):
        """`isometry_rs.synth_isometry` validates its own input shape directly, rather than
        relying solely on `Isometry.__init__` (tested above) to shield it: a non-power-of-2
        column count raises a `ValueError` instead of silently flooring to the nearest valid
        `m` and returning a circuit that does not implement the requested map.
        """
        from qiskit._accelerate import isometry as isometry_rs

        malformed = np.eye(4, dtype=complex)[:, :3]  # rows=4 (valid), cols=3 (invalid)
        with self.assertRaisesRegex(ValueError, "number of columns"):
            isometry_rs.synth_isometry(malformed, 0, 0, 1e-10)

    def test_synth_isometry_rejects_non_power_of_two_rows(self):
        """Same validation as `test_synth_isometry_rejects_non_power_of_two_columns`, for the
        row count instead of the column count.
        """
        from qiskit._accelerate import isometry as isometry_rs

        malformed = np.eye(4, dtype=complex)[:3, :2]  # rows=3 (invalid), cols=2 (valid)
        with self.assertRaisesRegex(ValueError, "number of rows"):
            isometry_rs.synth_isometry(malformed, 0, 0, 1e-10)

    def test_synth_isometry_rejects_too_many_columns(self):
        """`isometry_rs.synth_isometry` also rejects `cols > rows` (both powers of 2) with a
        `ValueError`, rather than indexing out of bounds and panicking: without this check, PyO3
        converts the panic into `pyo3_runtime.PanicException`, which (unlike a normal
        `ValueError`) subclasses `BaseException` directly rather than `Exception`, so code that
        defensively catches `Exception` around this call would not catch it.
        """
        from qiskit._accelerate import isometry as isometry_rs

        too_many_columns = np.eye(2, 4, dtype=complex)  # rows=2, cols=4: m > n
        with self.assertRaisesRegex(ValueError, "more columns.*than rows"):
            isometry_rs.synth_isometry(too_many_columns, 0, 0, 1e-10)


if __name__ == "__main__":
    unittest.main()
