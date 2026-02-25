# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A Test for Pauli Product Measurement instruction."""

import io
import numpy as np

from ddt import ddt, data
from qiskit.circuit import QuantumCircuit, Parameter, CircuitError, CommutationChecker
from qiskit.circuit.library import PauliRotationGate, PauliEvolutionGate, PauliProductMeasurement
from qiskit.quantum_info import Pauli, Operator, SparsePauliOp
from qiskit.qpy import dump, load
from qiskit.compiler import transpile
from test import QiskitTestCase  # pylint: disable=wrong-import-order


@ddt
class TestPauliRotationGate(QiskitTestCase):
    """Tests for the PauliRotationGate."""

    def test_simple(self):
        """Test a simple rotation gate."""
        pauli = Pauli("XIZZY")
        angle = 0.23
        rotation = PauliRotationGate(pauli, angle)

        expected = QuantumCircuit(pauli.num_qubits)
        expected.h(4)
        expected.sx(0)
        expected.cx(4, 2)
        expected.cx(2, 1)
        expected.cx(1, 0)
        expected.rz(angle, 0)
        expected.cx(1, 0)
        expected.cx(2, 1)
        expected.cx(4, 2)
        expected.sxdg(0)
        expected.h(4)

        with self.subTest(msg="decomposition"):
            self.assertEqual(expected, rotation.definition)

        with self.subTest(msg="matrix"):
            np.testing.assert_allclose(rotation.to_matrix(), Operator(rotation.definition).data)

    def test_equality(self):
        """Test some equalities."""
        x = Parameter("x")
        self.assertEqual(PauliRotationGate(Pauli("X"), 0.1), PauliRotationGate(Pauli("X"), 0.1))
        self.assertEqual(PauliRotationGate(Pauli("X"), x), PauliRotationGate(Pauli("X"), x))
        self.assertEqual(PauliRotationGate(Pauli("X"), -x), PauliRotationGate(Pauli("-X"), x))

        self.assertNotEqual(PauliRotationGate(Pauli("X"), 0.1), PauliRotationGate(Pauli("X"), 0.2))
        self.assertNotEqual(PauliRotationGate(Pauli("Y"), 0.1), PauliRotationGate(Pauli("X"), 0.1))

    @data("iX", "-iX")
    def test_invalid_phase(self, pauli):
        """Test invalid Pauli phases raises an error."""
        with self.assertRaises(CircuitError):
            _ = PauliRotationGate(Pauli(pauli), 1.0)

    def test_commutation_checks(self):
        """Test commutative optimization handles the rotation gate."""
        cc = CommutationChecker()
        xx = PauliRotationGate(Pauli("XX"), 0.1)
        yy = PauliRotationGate(Pauli("YY"), -0.5)
        self.assertTrue(cc.commute(xx, [0, 1], [], yy, [0, 1], []))
        self.assertTrue(cc.commute(xx, [1, 0], [], yy, [0, 1], []))
        self.assertFalse(cc.commute(xx, [2, 0], [], yy, [0, 1], []))

        zzm = PauliProductMeasurement(Pauli("ZZ"))
        self.assertTrue(cc.commute(zzm, [0, 1], [0], xx, [1, 0], []))
        self.assertFalse(cc.commute(zzm, [0, 2], [0], xx, [1, 0], []))

        summed = SparsePauliOp(["XXI", "IXI", "IIZ"])
        evo = PauliEvolutionGate(summed, time=42.2)
        self.assertTrue(cc.commute(evo, [0, 1, 2], [], xx, [1, 2], []))
        self.assertFalse(cc.commute(evo, [0, 1, 2], [], xx, [0, 1], []))
        self.assertFalse(cc.commute(evo, [0, 1, 2], [], yy, [0, 1], []))

    def test_pauli_evolution_equivalence(self):
        """Test consistency with the PauliEvolutionGate."""
        pauli = Pauli("XIYIZ")
        angle = Parameter("theta")
        evo = PauliEvolutionGate(pauli, angle / 2)
        rotation = PauliRotationGate(pauli, angle)

        self.assertEqual(evo.definition, rotation.definition)

    def test_python_rust_equivalence(self):
        """Test that the Python definition and Rust HLS definition match."""
        pauli = Pauli("IXYZ")
        angle = Parameter("theta")
        circuit = QuantumCircuit(pauli.num_qubits)
        circuit.append(PauliRotationGate(pauli, angle), circuit.qubits)

        # trigger Rust-path, HLS decomposition
        basis_gates = ["h", "sx", "sxdg", "rz", "cx"]
        rust_path = transpile(circuit, basis_gates=basis_gates, optimization_level=0)

        # compare to Python-side decomposition
        python_path = circuit.decompose()

        self.assertEqual(rust_path, python_path)

    def test_qpy(self):
        """Test qpy for circuits with PauliProductMeasurement instructions."""
        qc = QuantumCircuit(6, 2)
        x = Parameter("x")
        # qc.append(PauliProductMeasurement(Pauli("XZ")), [4, 1], [0])
        qc.append(PauliRotationGate(Pauli("XZ"), 0.2), [4, 1])
        qc.append(PauliRotationGate(Pauli("Z"), -12, label="wohooo rotation"), [2])
        qc.append(PauliRotationGate(Pauli("ZZ"), x), [3, 2])

        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circuit = load(qpy_file)[0]
        self.assertEqual(qc, new_circuit)

    def test_pauli_accessor(self):
        """Check that ``pauli()`` returns the original Pauli."""
        pauli = Pauli("XIYZ")
        rotation = PauliRotationGate(pauli, 0.0)
        self.assertEqual(rotation.pauli(), pauli)

    def test_draw(self):
        """Test drawing a Pauli rotation gate circuit."""
        qc = QuantumCircuit(3)
        qc.append(PauliRotationGate(Pauli("X"), 0.1), [0])
        qc.append(PauliRotationGate(Pauli("YY"), np.pi / 2), [1, 2])
        qc.append(PauliRotationGate(Pauli("ZZZ"), -np.pi / 4), [0, 1, 2])
        out = str(qc.draw("text"))
        expected = "\n".join(
            [
                "      ┌──────────┐ ┌──────────────┐",
                "q_0: ─┤ R_X(0.1) ├─┤0             ├",
                "     ┌┴──────────┴┐│              │",
                "q_1: ┤0           ├┤1 R_ZZZ(-π/4) ├",
                "     │  R_YY(π/2) ││              │",
                "q_2: ┤1           ├┤2             ├",
                "     └────────────┘└──────────────┘",
            ]
        )
        self.assertEqual(expected, out)
