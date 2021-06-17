# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Tests the implementation of parametric circuit.
"""
# TODO: remove print("\n{:s}\n{:s}\n{:s}\n".format("@" * 80, __doc__, "@" * 80))

import sys
import unittest
from collections import OrderedDict

# if os.getcwd() not in sys.path:
#     sys.path.append(os.getcwd())
import numpy as np

# TODO: remove parallelization!
from joblib import Parallel, delayed

from qiskit.test import QiskitTestCase
from qiskit.transpiler.synthesis.aqc.parametric_circuit import ParametricCircuit
from qiskit.transpiler.synthesis.aqc.utils import compare_circuits


class TestParametricCircuit(QiskitTestCase):
    """Tests ParametricCircuit."""

    def _matrix_conversion(self, num_qubits: int, depth: int) -> (int, float):
        print(".", end="", flush=True)
        self.assertTrue(isinstance(num_qubits, (int, np.int64)))
        self.assertTrue(isinstance(depth, (int, np.int64)))

        circuit = ParametricCircuit(
            num_qubits=num_qubits, layout="spin", connectivity="full", depth=depth
        )

        thetas = np.random.rand(circuit.num_thetas) * (2.0 * np.pi)
        circuit.set_thetas(thetas)

        residual = compare_circuits(
            target_circuit=circuit.to_numpy(), approx_circuit=circuit.to_qiskit(reverse=False)
        )
        self.assertLess(
            residual, float(np.sqrt(np.finfo(np.float64).eps)), "too big relative residual"
        )

        # Important when run inside a parallel process:
        sys.stderr.flush()
        sys.stdout.flush()

        return int(num_qubits), float(residual)

    def test_matrix_conversion(self):
        """Tests matrix conversion."""
        print("\nRunning {:s}() ...".format(self.test_matrix_conversion.__name__))
        print("Here we test that Numpy and Qiskit representations of")
        print("parametric circuit yield the same matrix.")

        nL = [(n, L) for n in range(2, 8 + 1) for L in range(10, 100)]
        results = Parallel(n_jobs=-1, prefer="processes")(
            delayed(self._matrix_conversion)(n, L) for n, L in nL
        )
        print("")
        sys.stderr.flush()
        sys.stdout.flush()

        # Group all the results by the number of qubits.
        d = OrderedDict()
        for nqubits, residual in results:
            d.setdefault(nqubits, []).append(residual)
        # Print out the maximal residual per number of qubits.
        for nqubits, residuals in d.items():
            print(
                "#qubits: {:2d}, maximal relative residual: {:0.16f}".format(
                    nqubits, max(residuals)
                )
            )

    def test_basic_functions(self):
        """Tests basic functions."""
        print("\nRunning {:s}() ...".format(self.test_basic_functions.__name__))
        print("Here we test the basic functionality of parametric circuit")

        eps = 10.0 * np.finfo(np.float64).eps

        for num_qubits in range(2, 8 + 1):
            for L in np.random.permutation(np.arange(10, 100))[0:10]:
                print(".", end="", flush=True)
                circuit = ParametricCircuit(
                    num_qubits=num_qubits, layout="spin", connectivity="full", depth=L
                )

                thetas = np.random.rand(circuit.num_thetas) * (2.0 * np.pi)
                circuit.set_thetas(thetas)
                self.assertTrue(np.allclose(thetas, circuit.thetas, atol=eps, rtol=eps))

                self.assertEqual(circuit.cnots.shape, (2, circuit.num_cnots))

                # pylint: disable=misplaced-comparison-constant
                self.assertTrue(np.all(1 <= circuit.cnots))
                self.assertTrue(np.all(circuit.cnots <= circuit.num_qubits))
        print("")


if __name__ == "__main__":
    np.set_printoptions(precision=6, linewidth=256)
    unittest.main()
