# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for visualization of circuit with Latex drawer."""

import unittest

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.visualization import circuit_drawer
from .visualization import QiskitVisualizationTestCase


class TestLatexSourceGenerator(QiskitVisualizationTestCase):
    """Qiskit latex source generator tests."""

    def test_empty_circuit(self):
        """Test draw an empty circuit"""
        filename = self._get_resource_path('test_empty.tex')
        qc = QuantumCircuit(1)
        circuit_drawer(qc, filename=filename, output='latex_source')

        self.assertEqualToReference(filename)

    def test_tiny_circuit(self):
        """Test draw tiny circuit."""
        filename = self._get_resource_path('test_tiny.tex')
        qc = QuantumCircuit(1)
        qc.h(0)

        circuit_drawer(qc, filename=filename, output='latex_source')

        self.assertEqualToReference(filename)

    def test_normal_circuit(self):
        """Test draw normal size circuit."""
        filename = self._get_resource_path('test_normal.tex')
        qc = QuantumCircuit(5)
        for qubit in range(5):
            qc.h(qubit)

        circuit_drawer(qc, filename=filename, output='latex_source')

        self.assertEqualToReference(filename)

    def test_4597(self):
        """Test cregbundle and conditional gates.
        See: https://github.com/Qiskit/qiskit-terra/pull/4597 """
        filename = self._get_resource_path('test_4597.tex')
        qr = QuantumRegister(3, 'q')
        cr = ClassicalRegister(3, 'c')
        qc = QuantumCircuit(qr, cr)
        qc.x(qr[2]).c_if(cr, 2)
        qc.draw(output='latex_source', cregbundle=True)

        circuit_drawer(qc, filename=filename, output='latex_source')

        self.assertEqualToReference(filename)

    def test_deep_circuit(self):
        """Test draw deep circuit."""
        filename = self._get_resource_path('test_deep.tex')
        qc = QuantumCircuit(1)
        for _ in range(100):
            qc.h(0)

        circuit_drawer(qc, filename=filename, output='latex_source')

        self.assertEqualToReference(filename)

    def test_huge_circuit(self):
        """Test draw huge circuit."""
        filename = self._get_resource_path('test_huge.tex')
        qc = QuantumCircuit(40)
        for qubit in range(39):
            qc.h(qubit)
            qc.cx(qubit, 39)

        circuit_drawer(qc, filename=filename, output='latex_source')

        self.assertEqualToReference(filename)

    def test_teleport(self):
        """Test draw teleport circuit."""
        from qiskit.circuit.library import U3Gate
        filename = self._get_resource_path('test_teleport.tex')
        qr = QuantumRegister(3, 'q')
        cr = ClassicalRegister(3, 'c')
        qc = QuantumCircuit(qr, cr)
        # Prepare an initial state
        qc.append(U3Gate(0.3, 0.2, 0.1), [qr[0]])
        # Prepare a Bell pair
        qc.h(qr[1])
        qc.cx(qr[1], qr[2])
        # Barrier following state preparation
        qc.barrier(qr)
        # Measure in the Bell basis
        qc.cx(qr[0], qr[1])
        qc.h(qr[0])
        qc.measure(qr[0], cr[0])
        qc.measure(qr[1], cr[1])
        # Apply a correction
        qc.z(qr[2]).c_if(cr, 1)
        qc.x(qr[2]).c_if(cr, 2)
        qc.measure(qr[2], cr[2])

        circuit_drawer(qc, filename=filename, output='latex_source')

        self.assertEqualToReference(filename)

    def test_global_phase(self):
        """Test circuit with global phase"""
        filename = self._get_resource_path('test_global_phase.tex')
        circuit = QuantumCircuit(3, global_phase=1.57079632679)
        circuit.h(range(3))

        circuit_drawer(circuit, filename=filename, output='latex_source')

        self.assertEqualToReference(filename)


if __name__ == '__main__':
    unittest.main(verbosity=2)
