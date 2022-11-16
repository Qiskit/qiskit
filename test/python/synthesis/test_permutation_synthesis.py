# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test permutation synthesis functions."""


import unittest
import numpy as np
from ddt import ddt, data

from qiskit.circuit.library import LinearFunction
from qiskit.synthesis.permutation import synth_permutation_depth_lnn_kms
from qiskit.synthesis.permutation.permutation_utils import _get_ordered_swap
from qiskit.test import QiskitTestCase


@ddt
class TestPermutationSynthesis(QiskitTestCase):
    """Test the permutation synthesis functions."""

    @data(4, 5, 10, 15, 20)
    def test_get_ordered_swap(self, width):
        """Test get_ordered_swap function produces correct swap list."""
        np.random.seed(1)
        for _ in range(5):
            permutation = np.random.permutation(width)
            swap_list = _get_ordered_swap(permutation)
            output = list(range(width))
            for i, j in swap_list:
                output[i], output[j] = output[j], output[i]
            self.assertTrue(np.array_equal(permutation, output))
            self.assertLess(len(swap_list), width)

    @data(4, 5, 10, 15, 20)
    def test_synth_permutation_depth_lnn_kms(self, width):
        """Test synth_permutation_depth_lnn_kms function produces the correct
        circuit."""
        np.random.seed(1)
        for _ in range(5):
            pattern = np.random.permutation(width)
            qc = synth_permutation_depth_lnn_kms(pattern)

            # Check that the synthesized circuit consists of SWAP gates only,
            # and that these SWAPs adhere to the LNN connectivity.
            for instruction in qc.data:
                self.assertEqual(instruction.operation.name, "swap")
                q0 = qc.find_bit(instruction.qubits[0]).index
                q1 = qc.find_bit(instruction.qubits[1]).index
                dist = abs(q0 - q1)
                self.assertEqual(dist, 1)

            # Check that the depth of the circuit (measured in #SWAPs)
            # does not exceed the number of qubits.
            self.assertLessEqual(qc.depth(), width)

            # Construct a linear function from the synthesized circuit, and
            # check that its permutation pattern matches the original pattern.
            synthesized_pattern = LinearFunction(qc).permutation_pattern()
            self.assertTrue(np.array_equal(synthesized_pattern, pattern))


if __name__ == "__main__":
    unittest.main()
