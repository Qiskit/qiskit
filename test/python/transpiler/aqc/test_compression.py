"""
Tests the original circuit vs compressed one for equivalence.
"""
print("\n{:s}\n{:s}\n{:s}\n".format("@" * 80, __doc__, "@" * 80))

import sys, os, traceback

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
import numpy as np
import unittest
from joblib import Parallel, delayed
from collections import OrderedDict
from qiskit.transpiler.synthesis.aqc.parametric_circuit import ParametricCircuit
from qiskit.transpiler.synthesis.aqc.compressor import EulerCompressor
from qiskit.transpiler.synthesis.aqc.utils import compare_circuits

# Avoid excessive deprecation warnings in Qiskit on Linux system.
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


class TestCompression(unittest.TestCase):
    def _euler_compression(self, nqubits: int, depth: int) -> (int, float):
        """
        Compares the original circuit vs compressed one for equivalence of
        underlying matrices.
        """
        print(".", end="", flush=True)
        self.assertTrue(isinstance(nqubits, (int, np.int64)))
        self.assertTrue(isinstance(depth, (int, np.int64)))

        circuit = ParametricCircuit(
            num_qubits=nqubits, layout="spin", connectivity="full", depth=depth
        )

        # Imitate group-lasso output by setting some thetas to zero.
        _NT = circuit.num_thetas_per_cnot
        thetas = np.random.rand(circuit.num_thetas) * (2.0 * np.pi)
        zeros = np.random.permutation(np.arange(depth))[0 : depth // 4]
        for z in zeros:
            thetas[_NT * z : _NT * (z + 1)] = 0.0
        circuit.set_thetas(thetas)

        compressed_circuit = EulerCompressor().compress(circuit)

        residual = compare_circuits(
            target_circuit=circuit.to_numpy(), approx_circuit=compressed_circuit.to_numpy()
        )
        self.assertLess(
            residual, float(np.sqrt(np.finfo(np.float64).eps)), "too big relative residual"
        )

        # Important when run inside a parallel process:
        sys.stderr.flush()
        sys.stdout.flush()
        return nqubits, residual

    def test_euler_compression(self):
        print("\nRunning {:s}() ...".format(self.test_euler_compression.__name__))
        print("Here we compare circuit matrices before and after compression.")
        print("They must be equal up to a small round-off error.")

        nL = [(n, L) for n in range(2, 8 + 1) for L in range(10, 100)]
        results = Parallel(n_jobs=-1, prefer="processes")(
            delayed(self._euler_compression)(n, L) for n, L in nL
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


if __name__ == "__main__":
    np.set_printoptions(precision=6, linewidth=256)
    try:
        unittest.main()
    except Exception as ex:
        print("message length:", len(str(ex)))
        traceback.print_exc()
