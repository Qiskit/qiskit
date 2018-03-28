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
from qiskit.tools.qi.qi import state_fidelity
from .common import QiskitTestCase


class TestInitialize(QiskitTestCase):
    """QISKIT InitializeGate tests."""

    _desired_fidelity = 0.99

    def test_uniform_superposition(self):
        desired_vector = [0.5, 0.5, 0.5, 0.5]
        qp = QuantumProgram()
        qr = qp.create_quantum_register("qr", 2)
        cr = qp.create_classical_register("cr", 2)
        qc = qp.create_circuit("qc", [qr], [cr])
        qc.initialize(desired_vector, [qr[0], qr[1]])
        result = qp.execute(["qc"], backend='local_qasm_simulator', shots=1)
        quantum_state = result.get_data("qc")['quantum_state']
        fidelity = state_fidelity(quantum_state, desired_vector)
        self.assertGreater(
            fidelity, self._desired_fidelity,
            "Initializer has low fidelity {0:.2g}.".format(fidelity))

    def test_deterministic_state(self):
        desired_vector = [0, 1, 0, 0]
        qp = QuantumProgram()
        qr = qp.create_quantum_register("qr", 2)
        cr = qp.create_classical_register("cr", 2)
        qc = qp.create_circuit("qc", [qr], [cr])
        qc.initialize(desired_vector, [qr[0], qr[1]])
        result = qp.execute(["qc"], backend='local_qasm_simulator', shots=1)
        quantum_state = result.get_data("qc")['quantum_state']
        fidelity = state_fidelity(quantum_state, desired_vector)
        self.assertGreater(
            fidelity, self._desired_fidelity,
            "Initializer has low fidelity {0:.2g}.".format(fidelity))

    def test_bell_state(self):
        desired_vector = [1/math.sqrt(2), 0, 0, 1/math.sqrt(2)]
        qp = QuantumProgram()
        qr = qp.create_quantum_register("qr", 2)
        cr = qp.create_classical_register("cr", 2)
        qc = qp.create_circuit("qc", [qr], [cr])
        qc.initialize(desired_vector, [qr[0], qr[1]])
        result = qp.execute(["qc"], backend='local_qasm_simulator', shots=1)
        quantum_state = result.get_data("qc")['quantum_state']
        fidelity = state_fidelity(quantum_state, desired_vector)
        self.assertGreater(
            fidelity, self._desired_fidelity,
            "Initializer has low fidelity {0:.2g}.".format(fidelity))

    def test_ghz_state(self):
        desired_vector = [1/math.sqrt(2), 0, 0, 0, 0, 0, 0, 1/math.sqrt(2)]
        qp = QuantumProgram()
        qr = qp.create_quantum_register("qr", 3)
        cr = qp.create_classical_register("cr", 3)
        qc = qp.create_circuit("qc", [qr], [cr])
        qc.initialize(desired_vector, [qr[0], qr[1], qr[2]])
        result = qp.execute(["qc"], backend='local_qasm_simulator', shots=1)
        quantum_state = result.get_data("qc")['quantum_state']
        fidelity = state_fidelity(quantum_state, desired_vector)
        self.assertGreater(
            fidelity, self._desired_fidelity,
            "Initializer has low fidelity {0:.2g}.".format(fidelity))

    def test_single_qubit(self):
        desired_vector = [1/math.sqrt(3), math.sqrt(2)/math.sqrt(3)]
        qp = QuantumProgram()
        qr = qp.create_quantum_register("qr", 1)
        cr = qp.create_classical_register("cr", 1)
        qc = qp.create_circuit("qc", [qr], [cr])
        qc.initialize(desired_vector, [qr[0]])
        result = qp.execute(["qc"], backend='local_qasm_simulator', shots=1)
        quantum_state = result.get_data("qc")['quantum_state']
        fidelity = state_fidelity(quantum_state, desired_vector)
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
        qr = qp.create_quantum_register("qr", 3)
        cr = qp.create_classical_register("cr", 1)
        qc = qp.create_circuit("qc", [qr], [cr])
        qc.initialize(desired_vector, [qr[0], qr[1], qr[2]])
        result = qp.execute(["qc"], backend='local_qasm_simulator', shots=1)
        quantum_state = result.get_data("qc")['quantum_state']
        fidelity = state_fidelity(quantum_state, desired_vector)
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
        qr = qp.create_quantum_register("qr", 4)
        cr = qp.create_classical_register("cr", 4)
        qc = qp.create_circuit("qc", [qr], [cr])
        qc.initialize(desired_vector, [qr[0], qr[1], qr[2], qr[3]])
        result = qp.execute(["qc"], backend='local_qasm_simulator', shots=1)
        quantum_state = result.get_data("qc")['quantum_state']
        fidelity = state_fidelity(quantum_state, desired_vector)
        self.assertGreater(
            fidelity, self._desired_fidelity,
            "Initializer has low fidelity {0:.2g}.".format(fidelity))

    def test_malformed_amplitudes(self):
        desired_vector = [1/math.sqrt(3), math.sqrt(2)/math.sqrt(3), 0]
        qp = QuantumProgram()
        qr = qp.create_quantum_register("qr", 2)
        cr = qp.create_classical_register("cr", 2)
        qc = qp.create_circuit("qc", [qr], [cr])
        self.assertRaises(
            QISKitError,
            qc.initialize, desired_vector, [qr[0], qr[1]])

    def test_non_unit_probability(self):
        desired_vector = [1, 1]
        qp = QuantumProgram()
        qr = qp.create_quantum_register("qr", 2)
        cr = qp.create_classical_register("cr", 2)
        qc = qp.create_circuit("qc", [qr], [cr])
        self.assertRaises(
            QISKitError,
            qc.initialize, desired_vector, [qr[0], qr[1]])

    def test_initialize_middle_circuit(self):
        desired_vector = [0.5, 0.5, 0.5, 0.5]
        qp = QuantumProgram()
        qr = qp.create_quantum_register("qr", 2)
        cr = qp.create_classical_register("cr", 2)
        qc = qp.create_circuit("qc", [qr], [cr])
        qc.h(qr[0])
        qc.cx(qr[0], qr[1])
        qc.reset(qr[0])
        qc.reset(qr[1])
        qc.initialize(desired_vector, [qr[0], qr[1]])
        result = qp.execute(["qc"], backend='local_qasm_simulator', shots=1)
        quantum_state = result.get_data("qc")['quantum_state']
        fidelity = state_fidelity(quantum_state, desired_vector)
        self.assertGreater(
            fidelity, self._desired_fidelity,
            "Initializer has low fidelity {0:.2g}.".format(fidelity))

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
        qr = qp.create_quantum_register("qr", 4)
        cr = qp.create_classical_register("cr", 4)
        qc = qp.create_circuit("qc", [qr], [cr])
        qc.initialize(desired_vector, [qr[0], qr[1], qr[2], qr[3]])
        result = qp.execute(["qc"], backend='local_qasm_simulator', shots=1)
        quantum_state = result.get_data("qc")['quantum_state']
        fidelity = state_fidelity(quantum_state, desired_vector)
        self.assertGreater(
            fidelity, self._desired_fidelity,
            "Initializer has low fidelity {0:.2g}.".format(fidelity))


if __name__ == '__main__':
    unittest.main()
