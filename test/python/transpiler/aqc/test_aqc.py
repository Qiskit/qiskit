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
Tests AQC framework using hardcoded and randomly generated circuits.
"""
# import os
import sys
import unittest

# if os.getcwd() not in sys.path:
#     sys.path.append(os.getcwd())
import numpy as np

# TODO: remove parallelization!
from joblib import Parallel, delayed

from qiskit.test import QiskitTestCase

# TODO: remove print("\n{:s}\n{:s}\n{:s}\n".format("@" * 80, __doc__, "@" * 80))
from qiskit.transpiler.synthesis.aqc.aqc import AQC
from qiskit.transpiler.synthesis.aqc.cnot_structures import make_cnot_network
from qiskit.transpiler.synthesis.aqc.parametric_circuit import ParametricCircuit
from qiskit.transpiler.synthesis.aqc.utils import compare_circuits, random_special_unitary
from .test_sample_data import ORIGINAL_CIRCUIT, INITIAL_THETAS


class TestAqc(QiskitTestCase):
    """Main tests of approximate quantum compiler."""

    def setUp(self) -> None:
        self._maxiter = int(5e3)
        self._eta = 0.1  # .1 for n=3, .01 for n=5
        self._tol = 0.01
        self._eps = 0.0
        self._reg = 0.7
        self._group = True

    def test_aqc_hardcoded(self):
        """Tests AQC on a hardcoded circuit/matrix."""
        print("\nRunning {:s}() ...".format(self.test_aqc_hardcoded.__name__))
        print("Here we test approximation of hardcoded matrix")

        num_qubits = int(round(np.log2(np.array(ORIGINAL_CIRCUIT).shape[0])))
        cnots = make_cnot_network(
            num_qubits=num_qubits, network_layout="spin", connectivity_type="full", depth=0
        )

        aqc = AQC(
            method="nesterov",
            maxiter=self._maxiter,
            eta=self._eta,
            tol=self._tol,
            eps=self._eps,
            reg=self._reg,
            group=True,
        )

        optimized_circuit = aqc.compile_unitary(
            target_matrix=np.array(ORIGINAL_CIRCUIT),
            cnots=cnots,
            thetas0=np.array(INITIAL_THETAS),
        )

        err = compare_circuits(optimized_circuit.to_numpy(), np.array(ORIGINAL_CIRCUIT))
        print("Relative difference between target and approximated matrices: {:0.6}".format(err))
        self.assertTrue(err < 1e-3)

    def _aqc_random(self, nqubits: int, depth: int) -> (int, int, float, float):
        """
        Implements a single test with randomly generated target matrix
        given the number of qubits and circuit depth.
        """
        print(".", end="", flush=True)
        self.assertTrue(isinstance(nqubits, (int, np.int64)))
        self.assertTrue(isinstance(depth, (int, np.int64)))

        target_matrix = random_special_unitary(num_qubits=nqubits)
        aqc = AQC(
            method="nesterov",
            maxiter=self._maxiter,
            eta=self._eta,
            tol=self._tol,
            eps=self._eps,
            reg=self._reg,
            group=self._group,
        )

        # Build the initial circuit.
        circuit0 = ParametricCircuit(
            num_qubits=nqubits, layout="spin", connectivity="full", depth=depth
        )
        circuit0.set_thetas(np.random.rand(circuit0.num_thetas) * (2 * np.pi))

        # Difference between the target and current circuit at the beginning.
        diff_before = compare_circuits(
            target_circuit=target_matrix, approx_circuit=circuit0.to_qiskit(tol=self._tol)
        )

        # Optimize the initial circuit and get a new, optimized one.
        optimized_circuit = aqc.compile_unitary(
            target_matrix=target_matrix, cnots=circuit0.cnots, thetas0=circuit0.thetas
        )

        # Evaluate difference after optimization.
        diff_after = compare_circuits(
            target_circuit=target_matrix, approx_circuit=optimized_circuit.to_qiskit(tol=self._tol)
        )

        # Important when run inside a parallel process:
        sys.stderr.flush()
        sys.stdout.flush()

        return int(nqubits), int(depth), float(diff_before), float(diff_after)

    # @unittest.skip("temporary skipping of the long test")
    def test_aqc_random(self):
        """Tests AQC on random unitary matrices."""
        print("\nRunning {:s}() ...".format(self.test_aqc_random.__name__))
        print("Here we test approximate compiling for different")
        print("qubit numbers and circuit depths, using random target")
        print("matrix and initial thetas.")

        num_layers = [
            (n, L) for n in range(3, 7) for L in np.random.permutation(np.arange(10, 100))[0:10]
        ]

        results = Parallel(n_jobs=-1, prefer="processes")(
            delayed(self._aqc_random)(n, L) for n, L in num_layers
        )
        print("")
        sys.stderr.flush()
        sys.stdout.flush()

        # Print out the results.
        for num_qubits, depth, diff_before, diff_after in results:
            print(
                "#qubits: {:d}, circuit depth: {:d};  relative residual:  "
                "initial: {:0.4f}, optimized: {:0.4f}".format(
                    num_qubits, depth, diff_before, diff_after
                )
            )


if __name__ == "__main__":
    unittest.main()
