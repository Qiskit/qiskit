# -*- coding: utf-8 -*-

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

"""
Initialize test.
"""

import math
import unittest
import numpy as np

from qiskit import QiskitError
from qiskit import QuantumCircuit
from qiskit import QuantumRegister
from qiskit import ClassicalRegister
from qiskit import execute, BasicAer
from qiskit.quantum_info import state_fidelity
from qiskit.test import QiskitTestCase


class TestInitialize(QiskitTestCase):
    """Qiskit Initialize tests."""

    _desired_fidelity = 0.99

    def test_uniform_superposition(self):
        """Initialize a uniform superposition on 2 qubits."""
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
        """Initialize a computational-basis state |01> on 2 qubits."""
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
        """Initialize a Bell state on 2 qubits."""
        desired_vector = [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)]
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
        """Initialize a GHZ state on 3 qubits."""
        desired_vector = [1 / math.sqrt(2), 0, 0, 0, 0, 0, 0, 1 / math.sqrt(2)]
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

    def test_initialize_register(self):
        """Initialize one register out of two."""
        desired_vector = [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)]
        qr = QuantumRegister(2, "qr")
        qr2 = QuantumRegister(2, "qr2")
        qc = QuantumCircuit(qr, qr2)
        qc.initialize(desired_vector, qr)
        job = execute(qc, BasicAer.get_backend('statevector_simulator'))
        result = job.result()
        statevector = result.get_statevector()
        fidelity = state_fidelity(statevector, np.kron([1, 0, 0, 0], desired_vector))
        self.assertGreater(
            fidelity, self._desired_fidelity,
            "Initializer has low fidelity {0:.2g}.".format(fidelity))

    def test_initialize_one_by_one(self):
        """Initializing qubits individually into product state same as initializing the pair."""
        qubit_0_state = [1, 0]
        qubit_1_state = [1 / math.sqrt(2), 1 / math.sqrt(2)]
        qr = QuantumRegister(2, "qr")
        qc_a = QuantumCircuit(qr)
        qc_a.initialize(np.kron(qubit_1_state, qubit_0_state), qr)

        qc_b = QuantumCircuit(qr)
        qc_b.initialize(qubit_0_state, [qr[0]])
        qc_b.initialize(qubit_1_state, [qr[1]])

        job = execute([qc_a, qc_b], BasicAer.get_backend('statevector_simulator'))
        result = job.result()
        statevector_a = result.get_statevector(0)
        statevector_b = result.get_statevector(1)
        fidelity = state_fidelity(statevector_a, statevector_b)
        self.assertGreater(
            fidelity, self._desired_fidelity,
            "Initializer has low fidelity {0:.2g}.".format(fidelity))

    def test_single_qubit(self):
        """Initialize a single qubit to a weighted superposition state."""
        desired_vector = [1 / math.sqrt(3), math.sqrt(2) / math.sqrt(3)]
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
        """Initialize to a non-trivial 3-qubit state."""
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
        """Initialize to a non-trivial 4-qubit state."""
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
        """Initializing to a vector with 3 amplitudes fails."""
        desired_vector = [1 / math.sqrt(3), math.sqrt(2) / math.sqrt(3), 0]
        qr = QuantumRegister(2, "qr")
        qc = QuantumCircuit(qr)
        self.assertRaises(
            QiskitError,
            qc.initialize, desired_vector, [qr[0], qr[1]])

    def test_non_unit_probability(self):
        """Initializing to a vector with probabilities not summing to 1 fails."""
        desired_vector = [1, 1]
        qr = QuantumRegister(2, "qr")
        qc = QuantumCircuit(qr)
        self.assertRaises(
            QiskitError,
            qc.initialize, desired_vector, [qr[0], qr[1]])

    def test_wrong_vector_size(self):
        """Initializing to a vector with a size different to the qubit parameter length.
        See https://github.com/Qiskit/qiskit-terra/issues/2372 """
        qr = QuantumRegister(2)

        random_state = [
            1 / math.sqrt(4) * complex(0, 1),
            1 / math.sqrt(8) * complex(1, 0),
            0,
            1 / math.sqrt(8) * complex(1, 0),
            1 / math.sqrt(8) * complex(0, 1),
            0,
            1 / math.sqrt(4) * complex(1, 0),
            1 / math.sqrt(8) * complex(1, 0)]

        qc = QuantumCircuit(qr)

        self.assertRaises(QiskitError, qc.initialize, random_state, qr[0:2])

    def test_initialize_middle_circuit(self):
        """Reset + initialize gives the correct statevector."""
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

    def test_math_amplitudes(self):
        """Initialize to amplitudes given by math expressions"""
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
        """Combining two circuits containing initialize."""
        desired_vector_1 = [1.0 / math.sqrt(2), 1.0 / math.sqrt(2)]
        desired_vector_2 = [1.0 / math.sqrt(2), -1.0 / math.sqrt(2)]
        qr = QuantumRegister(1, "qr")
        cr = ClassicalRegister(1, "cr")
        qc1 = QuantumCircuit(qr, cr)
        qc1.initialize(desired_vector_1, [qr[0]])

        qc2 = QuantumCircuit(qr, cr)
        qc2.initialize(desired_vector_2, [qr[0]])

        job = execute(qc1 + qc2, BasicAer.get_backend('statevector_simulator'))
        result = job.result()
        quantum_state = result.get_statevector()
        fidelity = state_fidelity(quantum_state, desired_vector_2)
        self.assertGreater(
            fidelity, self._desired_fidelity,
            "Initializer has low fidelity {0:.2g}.".format(fidelity))

    def test_equivalence(self):
        """Test two similar initialize instructions evaluate to equal."""
        desired_vector = [0.5, 0.5, 0.5, 0.5]
        qr = QuantumRegister(2, "qr")

        qc1 = QuantumCircuit(qr, name='circuit')
        qc1.initialize(desired_vector, [qr[0], qr[1]])

        qc2 = QuantumCircuit(qr, name='circuit')
        qc2.initialize(desired_vector, [qr[0], qr[1]])

        self.assertEqual(qc1, qc2)


if __name__ == '__main__':
    unittest.main()
