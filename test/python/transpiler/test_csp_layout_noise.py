# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the CSPLayoutNoise pass"""

import unittest
from math import pi

from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.passes import CSPLayoutNoise
from qiskit.converters import circuit_to_dag
from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeBogota, FakeMelbourne


class TestCSPLayoutNoise(QiskitTestCase):
    """Tests the CSPLayoutNoise pass"""
    seed = 42

    def test_2q_circuit_5q_coupling(self):
        """ 2 qubits in Bogota, with noise
            0 - qr1 - qr2 - 3 - 4
        """
        qr = QuantumRegister(2, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[0])  # qr1 -> qr0
        dag = circuit_to_dag(circuit)

        backend = FakeBogota()
        coupling_map = CouplingMap(backend.configuration().coupling_map)
        backend_prop = backend.properties()

        pass_ = CSPLayoutNoise(coupling_map,
                               backend_prop=backend_prop,
                               strict_direction=True,
                               seed=self.seed)
        pass_.run(dag)
        layout = pass_.property_set['layout']

        self.assertEqual(layout[qr[0]], 1)
        self.assertEqual(layout[qr[1]], 2)
        self.assertEqual(pass_.property_set['CSPLayout_stop_reason'], 'solution found')

    def test_3q_circuit_5q_coupling(self):
        """ 3 qubits in Bogota, with noise
            qr0 - qr1 - qr2 - 3 - 4
        """
        qr = QuantumRegister(3, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])  # qr1 -> qr0
        circuit.cx(qr[1], qr[2])  # qr0 -> qr2
        dag = circuit_to_dag(circuit)

        backend = FakeBogota()
        coupling_map = CouplingMap(backend.configuration().coupling_map)
        backend_prop = backend.properties()

        pass_ = CSPLayoutNoise(coupling_map,
                               backend_prop=backend_prop,
                               strict_direction=True,
                               seed=self.seed)
        pass_.run(dag)
        layout = pass_.property_set['layout']

        self.assertEqual(layout[qr[0]], 0)
        self.assertEqual(layout[qr[1]], 1)
        self.assertEqual(layout[qr[2]], 2)
        self.assertEqual(pass_.property_set['CSPLayout_stop_reason'], 'solution found')

    def test_3q_circuit_16_coupling(self):
        """ 3 qubits in Melbourne, with noise """
        qr = QuantumRegister(3, 'qr')
        cr = ClassicalRegister(3, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.x(qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.u1(0.2, qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[1], qr[2])
        circuit.u1(0.2, qr[2])
        circuit.cx(qr[1], qr[2])
        circuit.u3(0.2, pi/2, 3*pi/2, qr[0])
        circuit.u3(0.2, pi/2, 3*pi/2, qr[1])
        circuit.u3(0.2, pi/2, 3*pi/2, qr[2])
        circuit.measure(qr[0], cr[0])
        circuit.measure(qr[1], cr[1])
        circuit.measure(qr[2], cr[2])
        dag = circuit_to_dag(circuit)

        backend = FakeMelbourne()
        coupling_map = CouplingMap(backend.configuration().coupling_map)
        backend_prop = backend.properties()

        pass_ = CSPLayoutNoise(coupling_map,
                               backend_prop=backend_prop,
                               strict_direction=True,
                               seed=self.seed)
        pass_.run(dag)
        layout = pass_.property_set['layout']

        pass_old = CSPLayoutNoise(coupling_map,
                                  backend_prop=None,
                                  strict_direction=True,
                                  seed=self.seed)
        pass_old.run(dag)
        layout_old = pass_old.property_set['layout']

        self.assertNotEqual(layout, layout_old)

        self.assertEqual(layout[qr[0]], 11)
        self.assertEqual(layout[qr[1]], 12)
        self.assertEqual(layout[qr[2]], 2)
        self.assertEqual(pass_.property_set['CSPLayout_stop_reason'], 'solution found')


if __name__ == '__main__':
    unittest.main()
