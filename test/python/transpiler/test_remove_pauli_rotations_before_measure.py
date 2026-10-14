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

"""Test RemovePauliRotationsBeforeMeasure pass."""

import unittest

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import PauliProductMeasurement, PauliProductRotationGate
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import Pauli
from qiskit.transpiler.passes import RemovePauliRotationsBeforeMeasure

try:
    from test import QiskitTestCase
except (ImportError, AttributeError):
    from unittest import TestCase as QiskitTestCase


class TestRemovePauliRotationsBeforeMeasure(QiskitTestCase):
    """Test RemovePauliRotationsBeforeMeasure optimizations."""

    def test_remove_single_qubit_matching_ppr(self):
        """Remove a single matching 1Q PPR before PPM."""
        qr = QuantumRegister(1, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)
        circuit.append(PauliProductRotationGate(Pauli("Z"), 0.3), [qr[0]])
        circuit.append(PauliProductMeasurement(Pauli("Z")), [qr[0]], [cr[0]])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr, cr)
        expected.append(PauliProductMeasurement(Pauli("Z")), [qr[0]], [cr[0]])

        pass_ = RemovePauliRotationsBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_preserve_single_qubit_non_matching_ppr(self):
        """Do not remove a 1Q PPR with a different Pauli observable."""
        qr = QuantumRegister(1, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)
        circuit.append(PauliProductRotationGate(Pauli("X"), 0.3), [qr[0]])
        circuit.append(PauliProductMeasurement(Pauli("Z")), [qr[0]], [cr[0]])
        dag = circuit_to_dag(circuit)

        expected = circuit.copy()

        pass_ = RemovePauliRotationsBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_remove_multi_qubit_matching_ppr(self):
        """Remove a matching multi-qubit PPR before PPM."""
        qr = QuantumRegister(3, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)
        circuit.append(PauliProductRotationGate(Pauli("XYZ"), 0.5), qr)
        circuit.append(PauliProductMeasurement(Pauli("XYZ")), qr, [cr[0]])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr, cr)
        expected.append(PauliProductMeasurement(Pauli("XYZ")), qr, [cr[0]])

        pass_ = RemovePauliRotationsBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_preserve_commuting_different_multi_qubit_ppr(self):
        """Do not remove a commuting PPR if the physical observable differs."""
        # XX and ZZ commute on 2 qubits, but XX != ZZ.
        qr = QuantumRegister(2, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)
        circuit.append(PauliProductRotationGate(Pauli("XX"), 0.5), qr)
        circuit.append(PauliProductMeasurement(Pauli("ZZ")), qr, [cr[0]])
        dag = circuit_to_dag(circuit)

        expected = circuit.copy()

        pass_ = RemovePauliRotationsBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_preserve_subset_support_ppr(self):
        """Do not remove a PPR that acts on a strict subset of PPM qubits."""
        qr = QuantumRegister(2, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)
        circuit.append(PauliProductRotationGate(Pauli("Z"), 0.5), [qr[0]])
        circuit.append(PauliProductMeasurement(Pauli("ZZ")), [qr[0], qr[1]], [cr[0]])
        dag = circuit_to_dag(circuit)

        expected = circuit.copy()

        pass_ = RemovePauliRotationsBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_preserve_superset_support_ppr(self):
        """Do not remove a PPR if only a subset of its wires are measured."""
        qr = QuantumRegister(2, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)
        circuit.append(PauliProductRotationGate(Pauli("ZZ"), 0.5), qr)
        circuit.append(PauliProductMeasurement(Pauli("Z")), [qr[0]], [cr[0]])
        dag = circuit_to_dag(circuit)

        expected = circuit.copy()

        pass_ = RemovePauliRotationsBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_remove_permuted_qargs_matching_ppr(self):
        """Remove a matching multi-qubit PPR when qubits are specified in different order."""
        qr = QuantumRegister(2, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)
        # Pauli("XY") on [qr[0], qr[1]]: qr[0] is Y, qr[1] is X.
        circuit.append(PauliProductRotationGate(Pauli("XY"), 0.5), [qr[0], qr[1]])
        # Pauli("YX") on [qr[1], qr[0]]: qr[1] is X, qr[0] is Y.
        circuit.append(PauliProductMeasurement(Pauli("YX")), [qr[1], qr[0]], [cr[0]])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr, cr)
        expected.append(PauliProductMeasurement(Pauli("YX")), [qr[1], qr[0]], [cr[0]])

        pass_ = RemovePauliRotationsBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_preserve_ppr_with_additional_quantum_successor(self):
        """Do not remove a PPR if one of its output wires does not go to the PPM."""
        qr = QuantumRegister(2, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)
        circuit.append(PauliProductRotationGate(Pauli("ZZ"), 0.5), [qr[0], qr[1]])
        circuit.append(PauliProductMeasurement(Pauli("Z")), [qr[0]], [cr[0]])
        circuit.x(qr[1])
        dag = circuit_to_dag(circuit)

        expected = circuit.copy()

        pass_ = RemovePauliRotationsBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_remove_matching_ppr_with_parameterized_angle(self):
        """Remove a matching PPR even if its angle is a symbolic Parameter."""
        theta = Parameter("theta")
        qr = QuantumRegister(1, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)
        circuit.append(PauliProductRotationGate(Pauli("Z"), theta), [qr[0]])
        circuit.append(PauliProductMeasurement(Pauli("Z")), [qr[0]], [cr[0]])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr, cr)
        expected.append(PauliProductMeasurement(Pauli("Z")), [qr[0]], [cr[0]])

        pass_ = RemovePauliRotationsBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_remove_matching_ppr_with_negative_ppm_sign(self):
        """Remove a matching PPR when the PPM has a negative sign."""
        qr = QuantumRegister(1, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)
        circuit.append(PauliProductRotationGate(Pauli("Z"), 0.4), [qr[0]])
        circuit.append(PauliProductMeasurement(Pauli("-Z")), [qr[0]], [cr[0]])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr, cr)
        expected.append(PauliProductMeasurement(Pauli("-Z")), [qr[0]], [cr[0]])

        pass_ = RemovePauliRotationsBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_remove_consecutive_matching_pprs(self):
        """Remove consecutive matching PPRs before a PPM."""
        qr = QuantumRegister(1, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)
        circuit.append(PauliProductRotationGate(Pauli("Z"), 0.1), [qr[0]])
        circuit.append(PauliProductRotationGate(Pauli("Z"), 0.2), [qr[0]])
        circuit.append(PauliProductMeasurement(Pauli("Z")), [qr[0]], [cr[0]])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr, cr)
        expected.append(PauliProductMeasurement(Pauli("Z")), [qr[0]], [cr[0]])

        pass_ = RemovePauliRotationsBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_control_flow_recurse(self):
        """Ensure the pass recurses into control flow sub-blocks."""
        qr = QuantumRegister(1, "q")
        cr = ClassicalRegister(1, "c")
        sub_circ = QuantumCircuit(qr, cr)
        sub_circ.append(PauliProductRotationGate(Pauli("Z"), 0.5), [qr[0]])
        sub_circ.append(PauliProductMeasurement(Pauli("Z")), [qr[0]], [cr[0]])

        expected_sub_circ = QuantumCircuit(qr, cr)
        expected_sub_circ.append(PauliProductMeasurement(Pauli("Z")), [qr[0]], [cr[0]])

        main_circ = QuantumCircuit(qr, cr)
        main_circ.if_else((cr, 0), sub_circ, None, [qr[0]], [cr[0]])

        expected_main = QuantumCircuit(qr, cr)
        expected_main.if_else((cr, 0), expected_sub_circ, None, [qr[0]], [cr[0]])

        dag = circuit_to_dag(main_circ)
        pass_ = RemovePauliRotationsBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected_main), after)

    def test_preserve_identity_padded_support(self):
        """Do not remove a PPR with identity term if PPM does not share the same qubit support."""
        qr = QuantumRegister(2, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)
        circuit.append(PauliProductRotationGate(Pauli("ZI"), 0.5), [qr[0], qr[1]])
        circuit.append(PauliProductMeasurement(Pauli("Z")), [qr[0]], [cr[0]])
        dag = circuit_to_dag(circuit)

        expected = circuit.copy()

        pass_ = RemovePauliRotationsBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_pbc_pass_manager_integration(self):
        """Test integration in OptimizePBCPassManager stage plugin across levels 0, 1, 2, and 3."""
        from qiskit.transpiler.preset_passmanagers.pbc.plugins import (
            OptimizePBCPassManager,
            PassManagerPBCConfig,
        )

        qr = QuantumRegister(1, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)
        circuit.append(PauliProductRotationGate(Pauli("Z"), 0.3), [qr[0]])
        circuit.append(PauliProductMeasurement(Pauli("Z")), [qr[0]], [cr[0]])

        # Level 0 does not optimize
        pm_0 = OptimizePBCPassManager().pass_manager(PassManagerPBCConfig(), optimization_level=0)
        out_0 = pm_0.run(circuit)
        self.assertIn("pauli_product_rotation", [inst.operation.name for inst in out_0.data])

        # Level 1 includes RemovePauliRotationsBeforeMeasure
        pm_1 = OptimizePBCPassManager().pass_manager(PassManagerPBCConfig(), optimization_level=1)
        out_1 = pm_1.run(circuit)
        self.assertNotIn("pauli_product_rotation", [inst.operation.name for inst in out_1.data])
        self.assertIn("pauli_product_measurement", [inst.operation.name for inst in out_1.data])

        # Level 2 includes CommutativeOptimization and RemovePauliRotationsBeforeMeasure
        pm_2 = OptimizePBCPassManager().pass_manager(PassManagerPBCConfig(), optimization_level=2)
        out_2 = pm_2.run(circuit)
        self.assertNotIn("pauli_product_rotation", [inst.operation.name for inst in out_2.data])
        self.assertIn("pauli_product_measurement", [inst.operation.name for inst in out_2.data])

        # Level 3 includes CommutativeOptimization and RemovePauliRotationsBeforeMeasure
        pm_3 = OptimizePBCPassManager().pass_manager(PassManagerPBCConfig(), optimization_level=3)
        out_3 = pm_3.run(circuit)
        self.assertNotIn("pauli_product_rotation", [inst.operation.name for inst in out_3.data])
        self.assertIn("pauli_product_measurement", [inst.operation.name for inst in out_3.data])


if __name__ == "__main__":
    unittest.main()
