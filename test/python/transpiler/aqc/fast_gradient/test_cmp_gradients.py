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
Tests equivalence of the default and fast gradient computation routines.
"""
print("\n{:s}\n{:s}\n{:s}\n".format("@" * 80, __doc__, "@" * 80))

import sys, os

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
import time, traceback, gc
import numpy as np
import unittest
from joblib import Parallel, delayed
from qiskit.transpiler.synthesis.aqc.cnot_structures import generate_random_cnots
from qiskit.transpiler.synthesis.aqc.gradient import DefaultGradient
from qiskit.transpiler.synthesis.aqc.fast_gradient.fast_gradient import FastGradient
from qiskit.transpiler.synthesis.aqc.utils import random_SU

_WALL_TIME = 1
timer = time.perf_counter if _WALL_TIME != 0 else time.process_time


class TestCompareGradientImplementations(unittest.TestCase):
    def _compare(self, nqubits: int, depth: int) -> (int, int, float, float, float, float, float):
        """
        Calculates gradient and objective function value for the original
        and the fast implementations, and compares the outputs from both.
        Returns relative errors. Also, accumulates performance metrics.
        """
        print(".", end="", flush=True)
        self.assertTrue(isinstance(nqubits, (int, np.int64)))
        self.assertTrue(isinstance(depth, (int, np.int64)))
        _EPS = float(np.sqrt(np.finfo(np.float64).eps))
        _TINY = float(np.power(np.finfo(np.float64).tiny, 0.25))
        cnots = generate_random_cnots(num_qubits=nqubits, depth=depth)
        depth = cnots.shape[1]  # might be less than initial depth
        U = random_SU(nqubits)
        grad_dflt = DefaultGradient(num_qubits=nqubits, cnots=cnots)
        grad_fast = FastGradient(num_qubits=nqubits, cnots=cnots)
        thetas = np.random.rand(4 * depth + 3 * nqubits) * 2 * np.pi
        thetas = thetas.astype(np.float64)

        # Compute fast gradient. Disable garbage collection for a while.
        gc_enabled = gc.isenabled()
        gc.disable()
        start = timer()
        f1, g1 = grad_fast.get_gradient(thetas=thetas, target_matrix=U)
        t1 = timer() - start
        if gc_enabled:
            gc.enable()

        # Compute default gradient. Disable garbage collection for a while.
        gc_enabled = gc.isenabled()
        gc.disable()
        start = timer()
        f2, g2 = grad_dflt.get_gradient(thetas=thetas, target_matrix=U)
        t2 = timer() - start
        if gc_enabled:
            gc.enable()

        fobj_rel_err = abs(f1 - f2) / f2
        grad_rel_err = np.linalg.norm(g1 - g2) / np.linalg.norm(g2)
        speedup = t2 / max(t1, _TINY)
        self.assertLess(fobj_rel_err, _EPS)
        self.assertLess(grad_rel_err, _EPS)
        self.assertTrue(np.allclose(g1, g2, atol=_EPS, rtol=_EPS))

        # Important when run inside a parallel process:
        sys.stderr.flush()
        sys.stdout.flush()
        return int(nqubits), int(depth), fobj_rel_err, grad_rel_err, speedup, t1, t2

    def test_cmp_gradients(self):
        print("\nRunning {:s}() ...".format(self.test_cmp_gradients.__name__))
        print("Here we compare the default and fast gradient calculators")
        print("for equivalence")

        nL = [
            (n, L)
            for n in range(2, 8 + 1)
            for L in np.sort(
                np.random.permutation(np.arange(3 if n <= 3 else 7, 15 if n <= 3 else 100))[0:10]
            )
        ]

        _debug = False
        if _debug:
            results = list()
            for n, L in nL:
                results.append(self._compare(n, L))
        else:
            results = Parallel(n_jobs=-1, prefer="processes")(
                delayed(self._compare)(n, L) for n, L in nL
            )
        print("")
        sys.stderr.flush()
        sys.stdout.flush()

        # Print out the comparison results.
        total_speed_score, total_count = 0.0, 0
        for nqubits, depth, ferr, gerr, speedup, t1, t2 in results:
            total_speed_score += speedup
            total_count += 1
            print(
                "n: {:2d}, L: {:2d}, "
                "objective relative error: {:.16f}, "
                "gradient relative error: {:.16f}, "
                "speedup: {:6.3f},   "
                "'fast' time: {:6.3f}, 'default' time: {:6.3f}".format(
                    nqubits, depth, ferr, gerr, speedup, t1, t2
                )
            )
        print("\nTotal speedup score: {:0.6f}".format(total_speed_score))
        print("Mean speedup score: {:0.6f}".format(total_speed_score / total_count))


if __name__ == "__main__":
    np.set_printoptions(precision=6, linewidth=256)
    try:
        unittest.main()
    except Exception as ex:
        print("message length:", len(str(ex)))
        traceback.print_exc()
