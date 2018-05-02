# -*- coding: utf-8 -*-
# pylint: disable=invalid-name,missing-docstring

# Copyright 2017 IBM RESEARCH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""
InitializeGate (CompositeGate instance) test.
"""

import math
import unittest

from qiskit import QISKitError
from qiskit import QuantumProgram
from qiskit import QuantumCircuit
from qiskit import QuantumRegister
from qiskit import ClassicalRegister
from qiskit.tools.qi.qi import state_fidelity
from .common import QiskitTestCase


class TestInitialize(QiskitTestCase):
    """QISKIT InitializeGate tests."""

    _desired_fidelity = 0.99

    def test_uniform_superposition(self):
        desired_vector = [0.5, 0.5, 0.5, 0.5]
        qp = QuantumProgram()
        qr = QuantumRegister(2, "qr")
        qc = QuantumCircuit(qr)
        qc.initialize(desired_vector, [qr[0], qr[1]])
        qp.add_circuit("qc", qc)
        result = qp.execute("qc", backend='local_statevector_simulator')
        statevector = result.get_statevector()
        fidelity = state_fidelity(statevector, desired_vector)
        self.assertGreater(
            fidelity, self._desired_fidelity,
            "Initializer has low fidelity {0:.2g}.".format(fidelity))

    def test_deterministic_state(self):
        desired_vector = [0, 1, 0, 0]
        qp = QuantumProgram()
        qr = QuantumRegister(2, "qr")
        qc = QuantumCircuit(qr)
        qc.initialize(desired_vector, [qr[0], qr[1]])
        qp.add_circuit("qc", qc)
        result = qp.execute("qc", backend='local_statevector_simulator')
        statevector = result.get_statevector()
        fidelity = state_fidelity(statevector, desired_vector)
        self.assertGreater(
            fidelity, self._desired_fidelity,
            "Initializer has low fidelity {0:.2g}.".format(fidelity))

    def test_bell_state(self):
        desired_vector = [1/math.sqrt(2), 0, 0, 1/math.sqrt(2)]
        qp = QuantumProgram()
        qr = QuantumRegister(2, "qr")
        qc = QuantumCircuit(qr)
        qc = qp.create_circuit("qc", [qr])
        qc.initialize(desired_vector, [qr[0], qr[1]])
        qp.add_circuit("qc", qc)
        result = qp.execute("qc", backend='local_statevector_simulator')
        statevector = result.get_statevector()
        fidelity = state_fidelity(statevector, desired_vector)
        self.assertGreater(
            fidelity, self._desired_fidelity,
            "Initializer has low fidelity {0:.2g}.".format(fidelity))

    def test_ghz_state(self):
        desired_vector = [1/math.sqrt(2), 0, 0, 0, 0, 0, 0, 1/math.sqrt(2)]
        qp = QuantumProgram()
        qr = QuantumRegister(3, "qr")
        qc = QuantumCircuit(qr)
        qc = qp.create_circuit("qc", [qr])
        qc.initialize(desired_vector, [qr[0], qr[1], qr[2]])
        qp.add_circuit("qc", qc)
        result = qp.execute("qc", backend='local_statevector_simulator')
        statevector = result.get_statevector()
        fidelity = state_fidelity(statevector, desired_vector)
        self.assertGreater(
            fidelity, self._desired_fidelity,
            "Initializer has low fidelity {0:.2g}.".format(fidelity))

    def test_single_qubit(self):
        desired_vector = [1/math.sqrt(3), math.sqrt(2)/math.sqrt(3)]
        qp = QuantumProgram()
        qr = QuantumRegister(1, "qr")
        qc = QuantumCircuit(qr)
        qc.initialize(desired_vector, [qr[0]])
        qp.add_circuit("qc", qc)
        result = qp.execute("qc", backend='local_statevector_simulator')
        statevector = result.get_statevector()
        fidelity = state_fidelity(statevector, desired_vector)
        self.assertGreater(
            fidelity, self._desired_fidelity,
            "Initializer has low fidelity {0:.2g}.".format(fidelity))

    def test_random_3qubit(self):
        desired_vector = [
            1 / math.sqrt(16) * complex(0, 1),
            1 / math.sqrt(8) * complex(1, 0),
            1 / math.sqrt(16) * complex(1, 1),
            0,
            0,
            1 / math.sqrt(8) * complex(1, 2),
            1 / math.sqrt(16) * complex(1, 0),
            0]
        qp = QuantumProgram()
        qr = QuantumRegister(3, "qr")
        qc = QuantumCircuit(qr)
        qc.initialize(desired_vector, [qr[0], qr[1], qr[2]])
        qp.add_circuit("qc", qc)
        result = qp.execute("qc", backend='local_statevector_simulator')
        statevector = result.get_statevector()
        fidelity = state_fidelity(statevector, desired_vector)
        self.assertGreater(
            fidelity, self._desired_fidelity,
            "Initializer has low fidelity {0:.2g}.".format(fidelity))

    def test_random_4qubit(self):
        desired_vector = [
            1 / math.sqrt(4) * complex(0, 1),
            1 / math.sqrt(8) * complex(1, 0),
            0,
            0,
            0,
            0,
            0,
            0,
            1 / math.sqrt(8) * complex(1, 0),
            1 / math.sqrt(8) * complex(0, 1),
            0,
            0,
            0,
            0,
            1 / math.sqrt(4) * complex(1, 0),
            1 / math.sqrt(8) * complex(1, 0)]
        qp = QuantumProgram()
        qr = QuantumRegister(4, "qr")
        qc = QuantumCircuit(qr)
        qc.initialize(desired_vector, [qr[0], qr[1], qr[2], qr[3]])
        qp.add_circuit("qc", qc)
        result = qp.execute("qc", backend='local_statevector_simulator')
        statevector = result.get_statevector()
        fidelity = state_fidelity(statevector, desired_vector)
        self.assertGreater(
            fidelity, self._desired_fidelity,
            "Initializer has low fidelity {0:.2g}.".format(fidelity))

    def test_malformed_amplitudes(self):
        desired_vector = [1/math.sqrt(3), math.sqrt(2)/math.sqrt(3), 0]
        qr = QuantumRegister(2, "qr")
        qc = QuantumCircuit(qr)
        self.assertRaises(
            QISKitError,
            qc.initialize, desired_vector, [qr[0], qr[1]])

    def test_non_unit_probability(self):
        desired_vector = [1, 1]
        qr = QuantumRegister(2, "qr")
        qc = QuantumCircuit(qr)
        self.assertRaises(
            QISKitError,
            qc.initialize, desired_vector, [qr[0], qr[1]])

    def test_initialize_middle_circuit(self):
        desired_vector = [0.5, 0.5, 0.5, 0.5]
        qp = QuantumProgram()
        qr = QuantumRegister(2, "qr")
        cr = ClassicalRegister(2, "cr")
        qc = QuantumCircuit(qr, cr)
        qc.h(qr[0])
        qc.cx(qr[0], qr[1])
        qc.reset(qr[0])
        qc.reset(qr[1])
        qc.initialize(desired_vector, [qr[0], qr[1]])
        qc.measure(qr, cr)
        qp.add_circuit("qc", qc)
        # statevector simulator does not support reset
        shots = 2000
        threshold = 0.04 * shots
        result = qp.execute("qc", backend='local_qasm_simulator', shots=shots)
        counts = result.get_counts()
        target = {'00': shots / 4, '01': shots / 4, '10': shots / 4, '11': shots / 4}
        self.assertDictAlmostEqual(counts, target, threshold)

    def test_sympy(self):
        desired_vector = [
            0,
            math.cos(math.pi / 3) * complex(0, 1) / math.sqrt(4),
            math.sin(math.pi / 3) / math.sqrt(4),
            0,
            0,
            0,
            0,
            0,
            1 / math.sqrt(8) * complex(1, 0),
            1 / math.sqrt(8) * complex(0, 1),
            0,
            0,
            0,
            0,
            1 / math.sqrt(4),
            1 / math.sqrt(4) * complex(0, 1)]
        qp = QuantumProgram()
        qr = QuantumRegister(4, "qr")
        qc = QuantumCircuit(qr)
        qc.initialize(desired_vector, [qr[0], qr[1], qr[2], qr[3]])
        qp.add_circuit("qc", qc)
        result = qp.execute("qc", backend='local_statevector_simulator')
        statevector = result.get_statevector()
        fidelity = state_fidelity(statevector, desired_vector)
        self.assertGreater(
            fidelity, self._desired_fidelity,
            "Initializer has low fidelity {0:.2g}.".format(fidelity))


if __name__ == '__main__':
    unittest.main()
