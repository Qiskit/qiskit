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

"""Test the linear amplitude function."""

import unittest
from functools import partial
from collections import defaultdict
from ddt import ddt, data, unpack
import numpy as np

from qiskit import BasicAer, execute
from qiskit.test.base import QiskitTestCase
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import LinearAmplitudeFunction


@ddt
class TestLinearAmplitudeFunctional(QiskitTestCase):
    """Test the functional Pauli rotations."""

    def assertFunctionIsCorrect(self, function_circuit, reference):
        """Assert that ``function_circuit`` implements the reference function ``reference``."""
        num_ancillas = function_circuit.num_ancillas
        num_state_qubits = function_circuit.num_qubits - num_ancillas - 1

        circuit = QuantumCircuit(function_circuit.num_qubits)
        circuit.h(list(range(num_state_qubits)))
        circuit.append(function_circuit.to_instruction(), list(range(circuit.num_qubits)))

        backend = BasicAer.get_backend('statevector_simulator')
        statevector = execute(circuit, backend).result().get_statevector()

        probabilities = defaultdict(float)
        for i, statevector_amplitude in enumerate(statevector):
            i = bin(i)[2:].zfill(circuit.num_qubits)[num_ancillas:]
            probabilities[i] += np.real(np.abs(statevector_amplitude) ** 2)

        unrolled_probabilities = []
        unrolled_expectations = []
        for i, probability in probabilities.items():
            x, last_qubit = int(i[1:], 2), i[0]
            if last_qubit == '0':
                expected_amplitude = np.cos(reference(x)) / np.sqrt(2**num_state_qubits)
            else:
                expected_amplitude = np.sin(reference(x)) / np.sqrt(2**num_state_qubits)

            unrolled_probabilities += [probability]
            unrolled_expectations += [np.real(np.abs(expected_amplitude) ** 2)]

        np.testing.assert_almost_equal(unrolled_probabilities, unrolled_expectations)

    def evaluate_function(self, x_int, num_qubits, slope, offset, domain, image, rescaling_factor,
                          breakpoints=None):
        """A helper function to get the expected value of the linear amplitude function."""
        a, b = domain
        c, d = image

        # map integer x to the domain of the function
        x = a + (b - a) / (2 ** num_qubits - 1) * x_int

        # apply the function
        if breakpoints is None:
            value = offset + slope * x
        else:
            value = 0
            for i, point in enumerate(reversed(breakpoints)):
                if x >= point:
                    value = offset[-(i + 1)] + slope[-(i + 1)] * (x - point)
                    break

        # map the value to [0, 1]
        normalized = (value - c) / (d - c)

        # prepared value for taylor approximation
        return np.pi / 4 + np.pi * rescaling_factor / 2 * (normalized - 0.5)

    @data(
        (2, 1, 0, (0, 3), (0, 3), 0.1, None),
        (3, 1, 0, (0, 1), (0, 1), 0.01, None),
        (1, [0, 0], [0, 0], (0, 2), (0, 1), 0.1, [0, 1]),
        (2, [1, -1], [0, 1], (0, 2), (0, 1), 0.1, [0, 1]),
        (3, [1, 0, -1, 0], [0, 0.5, -0.5, -0.5], (0, 2.5), (-0.5, 0.5), 0.1, [0, 0.5, 1, 2]),
    )
    @unpack
    def test_polynomial_function(self, num_state_qubits, slope, offset, domain, image,
                                 rescaling_factor, breakpoints):
        """Test the polynomial rotation."""
        reference = partial(self.evaluate_function,
                            num_qubits=num_state_qubits,
                            slope=slope,
                            offset=offset,
                            domain=domain,
                            image=image,
                            rescaling_factor=rescaling_factor,
                            breakpoints=breakpoints)

        linear_f = LinearAmplitudeFunction(num_state_qubits, slope, offset, domain, image,
                                           rescaling_factor, breakpoints)

        self.assertFunctionIsCorrect(linear_f, reference)

    def test_not_including_start_in_breakpoints(self):
        """Test not including the start of the domain works."""
        num_state_qubits = 1
        slope = [0, 0]
        offset = [0, 0]
        domain = (0, 2)
        image = (0, 1)
        rescaling_factor = 0.1
        breakpoints = [1]

        reference = partial(self.evaluate_function,
                            num_qubits=num_state_qubits,
                            slope=slope,
                            offset=offset,
                            domain=domain,
                            image=image,
                            rescaling_factor=rescaling_factor,
                            breakpoints=breakpoints)

        linear_f = LinearAmplitudeFunction(num_state_qubits, slope, offset, domain, image,
                                           rescaling_factor, breakpoints)

        self.assertFunctionIsCorrect(linear_f, reference)

    def test_invalid_inputs_raise(self):
        """Test passing invalid inputs to the LinearAmplitudeFunction raises an error."""
        # default values
        num_state_qubits = 1
        slope = [0, 0]
        offset = [0, 0]
        domain = (0, 2)
        image = (0, 1)
        rescaling_factor = 0.1
        breakpoints = [0, 1]

        with self.subTest('mismatching breakpoints size'):
            with self.assertRaises(ValueError):
                _ = LinearAmplitudeFunction(num_state_qubits, slope, offset, domain, image,
                                            rescaling_factor, [0])

        with self.subTest('mismatching offsets'):
            with self.assertRaises(ValueError):
                _ = LinearAmplitudeFunction(num_state_qubits, slope, [0], domain, image,
                                            rescaling_factor, breakpoints)

        with self.subTest('mismatching slopes'):
            with self.assertRaises(ValueError):
                _ = LinearAmplitudeFunction(num_state_qubits, [0], offset, domain, image,
                                            rescaling_factor, breakpoints)

        with self.subTest('breakpoints outside of domain'):
            with self.assertRaises(ValueError):
                _ = LinearAmplitudeFunction(num_state_qubits, slope, offset, (0, 0.2), image,
                                            rescaling_factor, breakpoints)

        with self.subTest('breakpoints not sorted'):
            with self.assertRaises(ValueError):
                _ = LinearAmplitudeFunction(num_state_qubits, slope, offset, domain, image,
                                            rescaling_factor, [1, 0])

    def test_post_processing(self):
        """Test the ``post_processing`` method."""
        num_state_qubits = 2
        slope = 1
        offset = -2
        domain = (0, 2)
        image = (-2, 0)
        rescaling_factor = 0.1

        circuit = LinearAmplitudeFunction(num_state_qubits, slope, offset, domain, image,
                                          rescaling_factor)

        values = [0, 0.2, 0.5, 0.9, 1]

        def reference_post_processing(x):
            x = 2 / np.pi / rescaling_factor * (x - 0.5) + 0.5
            return image[0] + (image[1] - image[0]) * x

        expected = [reference_post_processing(value) for value in values]
        actual = [circuit.post_processing(value) for value in values]

        self.assertListEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()
