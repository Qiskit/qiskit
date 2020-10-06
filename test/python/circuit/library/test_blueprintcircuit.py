# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the blueprint circuit."""

import unittest

from qiskit.test.base import QiskitTestCase
from qiskit.circuit import QuantumRegister, Parameter, QuantumCircuit
from qiskit.circuit.library import BlueprintCircuit


class MockBlueprint(BlueprintCircuit):
    """A mock blueprint class."""

    def __init__(self, num_qubits):
        super().__init__(name='mock')
        self.num_qubits = num_qubits

    @property
    def num_qubits(self):
        return self._num_qubits

    @num_qubits.setter
    def num_qubits(self, num_qubits):
        self._invalidate()
        self._num_qubits = num_qubits
        self.qregs = [QuantumRegister(self.num_qubits, name='q')]

    def _check_configuration(self, raise_on_failure=True):
        valid = True
        if self.num_qubits is None:
            valid = False
            if raise_on_failure:
                raise AttributeError('The number of qubits was not set.')

        if self.num_qubits < 1:
            valid = False
            if raise_on_failure:
                raise ValueError('The number of qubits must at least be 1.')

        return valid

    def _build(self):
        super()._build()

        # pylint: disable=no-member
        self.rx(Parameter('angle'), 0)
        self.h(self.qubits)


class TestBlueprintCircuit(QiskitTestCase):
    """Test the blueprint circuit."""

    def test_invalidate_rebuild(self):
        """Test that invalidate and build reset and set _data and _parameter_table."""
        mock = MockBlueprint(5)
        mock._build()

        with self.subTest(msg='after building'):
            self.assertGreater(len(mock._data), 0)
            self.assertEqual(len(mock._parameter_table), 1)

        mock._invalidate()
        with self.subTest(msg='after invalidating'):
            self.assertTrue(mock._data is None)
            self.assertEqual(len(mock._parameter_table), 0)

        mock._build()
        with self.subTest(msg='after re-building'):
            self.assertGreater(len(mock._data), 0)
            self.assertEqual(len(mock._parameter_table), 1)

    def test_calling_attributes_works(self):
        """Test that the circuit is constructed when attributes are called."""
        properties = ['data']
        for prop in properties:
            with self.subTest(prop=prop):
                circuit = MockBlueprint(3)
                getattr(circuit, prop)
                self.assertGreater(len(circuit._data), 0)

        methods = ['qasm', 'count_ops', 'num_connected_components', 'num_nonlocal_gates',
                   'depth', '__len__', 'copy', 'inverse']
        for method in methods:
            with self.subTest(method=method):
                circuit = MockBlueprint(3)
                getattr(circuit, method)()
                self.assertGreater(len(circuit._data), 0)

        with self.subTest(method='__get__[0]'):
            circuit = MockBlueprint(3)
            _ = circuit[2]
            self.assertGreater(len(circuit._data), 0)

    def test_compose_works(self):
        """Test that the circuit is constructed when compose is called."""
        qc = QuantumCircuit(3)
        qc.x([0, 1, 2])
        circuit = MockBlueprint(3)
        circuit.compose(qc, inplace=True)

        reference = QuantumCircuit(3)
        reference.rx(list(circuit.parameters)[0], 0)
        reference.h([0, 1, 2])
        reference.x([0, 1, 2])

        self.assertEqual(reference, circuit)


if __name__ == '__main__':
    unittest.main()
