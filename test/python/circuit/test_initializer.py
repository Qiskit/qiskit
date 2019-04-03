# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=missing-docstring

"""
InitializeGate test.
"""

import math
import unittest

from qiskit import QiskitError
from qiskit import QuantumCircuit
from qiskit import QuantumRegister
from qiskit import ClassicalRegister
from qiskit import execute, BasicAer
from qiskit.quantum_info import state_fidelity
from qiskit.test import QiskitTestCase


class TestInitialize(QiskitTestCase):
    """Qiskit InitializeGate tests."""

    _desired_fidelity = 0.99

    def test_uniform_superposition(self):
        desired_vector = [0.5, 0.5, 0.5, 0.5]
        qr = QuantumRegister(2, "qr")
        qc = QuantumCircuit(qr)
        qc.initialize(desired_vector, [qr[0], qr[1]])
        job = execute(qc, BasicAer.get_backend('statevector_simulator'))
        result = job.result()
        statevector = result.get_statevector()
        fidelity = state_fidelity(statevector, desired_vector)
        self.assertGreater(
            fidelity, self._desired_fidelity,
            "Initializer has low fidelity {0:.2g}.".format(fidelity))

    def test_deterministic_state(self):
        desired_vector = [0, 1, 0, 0]
        qr = QuantumRegister(2, "qr")
        qc = QuantumCircuit(qr)
        qc.initialize(desired_vector, [qr[0], qr[1]])
        job = execute(qc, BasicAer.get_backend('statevector_simulator'))
        result = job.result()
        statevector = result.get_statevector()
        fidelity = state_fidelity(statevector, desired_vector)
        self.assertGreater(
            fidelity, self._desired_fidelity,
            "Initializer has low fidelity {0:.2g}.".format(fidelity))

    def test_bell_state(self):
        desired_vector = [1/math.sqrt(2), 0, 0, 1/math.sqrt(2)]
        qr = QuantumRegister(2, "qr")
        qc = QuantumCircuit(qr)
        qc.initialize(desired_vector, [qr[0], qr[1]])
        job = execute(qc, BasicAer.get_backend('statevector_simulator'))
        result = job.result()
        statevector = result.get_statevector()
        fidelity = state_fidelity(statevector, desired_vector)
        self.assertGreater(
            fidelity, self._desired_fidelity,
            "Initializer has low fidelity {0:.2g}.".format(fidelity))

    def test_ghz_state(self):
        desired_vector = [1/math.sqrt(2), 0, 0, 0, 0, 0, 0, 1/math.sqrt(2)]
        qr = QuantumRegister(3, "qr")
        qc = QuantumCircuit(qr)
        qc.initialize(desired_vector, [qr[0], qr[1], qr[2]])
        job = execute(qc, BasicAer.get_backend('statevector_simulator'))
        result = job.result()
        statevector = result.get_statevector()
        fidelity = state_fidelity(statevector, desired_vector)
        self.assertGreater(
            fidelity, self._desired_fidelity,
            "Initializer has low fidelity {0:.2g}.".format(fidelity))

    def test_single_qubit(self):
        desired_vector = [1/math.sqrt(3), math.sqrt(2)/math.sqrt(3)]
        qr = QuantumRegister(1, "qr")
        qc = QuantumCircuit(qr)
        qc.initialize(desired_vector, [qr[0]])
        job = execute(qc, BasicAer.get_backend('statevector_simulator'))
        result = job.result()
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
        qr = QuantumRegister(3, "qr")
        qc = QuantumCircuit(qr)
        qc.initialize(desired_vector, [qr[0], qr[1], qr[2]])
        job = execute(qc, BasicAer.get_backend('statevector_simulator'))
        result = job.result()
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
        qr = QuantumRegister(4, "qr")
        qc = QuantumCircuit(qr)
        qc.initialize(desired_vector, [qr[0], qr[1], qr[2], qr[3]])
        job = execute(qc, BasicAer.get_backend('statevector_simulator'))
        result = job.result()
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
            QiskitError,
            qc.initialize, desired_vector, [qr[0], qr[1]])

    def test_non_unit_probability(self):
        desired_vector = [1, 1]
        qr = QuantumRegister(2, "qr")
        qc = QuantumCircuit(qr)
        self.assertRaises(
            QiskitError,
            qc.initialize, desired_vector, [qr[0], qr[1]])

    def test_initialize_middle_circuit(self):
        desired_vector = [0.5, 0.5, 0.5, 0.5]
        qr = QuantumRegister(2, "qr")
        cr = ClassicalRegister(2, "cr")
        qc = QuantumCircuit(qr, cr)
        qc.h(qr[0])
        qc.cx(qr[0], qr[1])
        qc.reset(qr[0])
        qc.reset(qr[1])
        qc.initialize(desired_vector, [qr[0], qr[1]])
        qc.measure(qr, cr)
        # statevector simulator does not support reset
        shots = 2000
        threshold = 0.04 * shots
        job = execute(qc, BasicAer.get_backend('qasm_simulator'), shots=shots)
        result = job.result()
        counts = result.get_counts()
        target = {'00': shots / 4, '01': shots / 4,
                  '10': shots / 4, '11': shots / 4}
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
        qr = QuantumRegister(4, "qr")
        qc = QuantumCircuit(qr)
        qc.initialize(desired_vector, [qr[0], qr[1], qr[2], qr[3]])
        job = execute(qc, BasicAer.get_backend('statevector_simulator'))
        result = job.result()
        statevector = result.get_statevector()
        fidelity = state_fidelity(statevector, desired_vector)
        self.assertGreater(
            fidelity, self._desired_fidelity,
            "Initializer has low fidelity {0:.2g}.".format(fidelity))

    def test_combiner(self):
        desired_vector = [0, 1]
        qr = QuantumRegister(1, "qr")
        cr = ClassicalRegister(1, "cr")
        qc1 = QuantumCircuit(qr, cr)
        qc1.initialize([1.0 / math.sqrt(2), 1.0 / math.sqrt(2)], [qr[0]])

        qc2 = QuantumCircuit(qr, cr)
        qc2.initialize([1.0 / math.sqrt(2), -1.0 / math.sqrt(2)], [qr[0]])

        job = execute(qc1 + qc2, BasicAer.get_backend('statevector_simulator'))
        result = job.result()
        quantum_state = result.get_statevector()
        fidelity = state_fidelity(quantum_state, desired_vector)
        self.assertGreater(
            fidelity, self._desired_fidelity,
            "Initializer has low fidelity {0:.2g}.".format(fidelity))


if __name__ == '__main__':
    unittest.main()
