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
Tests correctness of compression algorithms for CNOT structures.
"""

# TODO: remove print("\n{:s}\n{:s}\n{:s}\n".format("@" * 80, __doc__, "@" * 80))

# import os
import sys
import unittest
from collections import OrderedDict

import numpy as np

# if os.getcwd() not in sys.path:
#     sys.path.append(os.getcwd())
from joblib import Parallel, delayed

from qiskit.test import QiskitTestCase
from qiskit.transpiler.synthesis.aqc.cnot_compress import (
    CNotCompressor,
    CNotSynthesis,
    compress_cnots,
)
from qiskit.transpiler.synthesis.aqc.cnot_structures import generate_random_cnots


class TestCNotCompress(QiskitTestCase):
    """Tests CNOT compression functions."""
    @staticmethod
    def _print_progress(cnot_reduction: int):
        if cnot_reduction > 0:
            print("+", end="", flush=True)
        elif cnot_reduction == 0:
            print(".", end="", flush=True)
        else:
            excess = abs(cnot_reduction)
            print("!" if excess > 9 else str(excess), end="", flush=True)

    def _synthesis(self, nqubits: int, depth: int) -> bool:
        """
        Runs Synthesis algorithm on a randomly generated CNOT structure.
        """
        cnots = generate_random_cnots(num_qubits=nqubits, depth=depth, set_depth_limit=False)
        self.assertTrue(depth == cnots.shape[1])
        synth_cnots = CNotSynthesis.synthesis(nqubits, cnots, False)
        delta = int(cnots.shape[1] - synth_cnots.shape[1])
        self._print_progress(delta)
        self.assertTrue(CNotSynthesis.compare_cnot_circuits(nqubits, cnots, synth_cnots))
        return True

    def _cnot_compressor(self, nqubits: int, depth: int) -> (int, int):
        """
        Applies CNOT compressor to a randomly generated CNOT structure.
        Returns:
            (number of qubits, reduction of CNOTs after compression)
        """
        cnots = generate_random_cnots(num_qubits=nqubits, depth=depth, set_depth_limit=False)
        self.assertTrue(depth == cnots.shape[1])
        compressed = CNotCompressor.compress(nqubits, cnots)
        if (depth % 5) == 0:
            print(".", end="", flush=True)
        self.assertTrue(compressed.size <= cnots.size)
        self.assertTrue(CNotSynthesis.compare_cnot_circuits(nqubits, cnots, compressed))
        return int(nqubits), int(cnots.shape[1] - compressed.shape[1])

    def _full_compressor(self, nqubits: int, depth: int) -> (int, int):
        """
        Runs CompressCNOTs() function, which combines rule-based and
        Synthesis algorithms.
        Returns:
            (number of qubits, reduction of CNOTs after compression)
        """
        cnots = generate_random_cnots(num_qubits=nqubits, depth=depth, set_depth_limit=False)
        self.assertTrue(depth == cnots.shape[1])
        compressed = compress_cnots(nqubits, cnots, False, False)
        delta = int(cnots.shape[1] - compressed.shape[1])
        if (depth % 5) == 0:
            self._print_progress(delta)
        self.assertTrue(CNotSynthesis.compare_cnot_circuits(nqubits, cnots, compressed))
        return int(nqubits), int(delta)

    # @unittest.skip("temporary skipping this test")
    def test_synthesis(self):
        """Tests synthesis."""
        print("\nRunning {:s}() ...".format(self.test_synthesis.__name__))
        print("Here we check that Synthesis algorithm does not distort")
        print("circuit matrix, but possibly changes the number of CNOTs.")

        nL = [
            (repeat, n, L)
            for repeat in range(5)
            for n in range(2, 8 + 1)
            for L in np.arange(1, 100)
        ]

        _debug = False
        if _debug:
            results = list()
            for _, n, L in nL:
                results.append(self._synthesis(n, L))
        else:
            results = Parallel(n_jobs=-1, prefer="processes")(
                delayed(self._synthesis)(n, L) for _, n, L in nL
            )
        print("")
        sys.stderr.flush()
        sys.stdout.flush()
        self.assertTrue(len(results) > 0)

    # @unittest.skip("temporary skipping this test")
    def test_cnot_compressor(self):
        """Tests CNOT compressor."""
        print("\nRunning {:s}() ...".format(self.test_cnot_compressor.__name__))
        print("Here we check that CNOT compressor does not distort")
        print("circuit matrix, but possibly reduces the number of CNOTs.")

        nL = [
            (repeat, n, L)
            for repeat in range(5)
            for n in range(2, 10 + 1)
            for L in np.arange(1, 100)
        ]

        _debug = False
        if _debug:
            results = list()
            for _, n, L in nL:
                results.append(self._cnot_compressor(n, L))
        else:
            results = Parallel(n_jobs=-1, prefer="processes")(
                delayed(self._cnot_compressor)(n, L) for _, n, L in nL
            )
        print("")
        sys.stderr.flush()
        sys.stdout.flush()

        # Group all the results by the number of qubits.
        d = OrderedDict()
        for nqubits, reduction in results:
            d.setdefault(nqubits, []).append(reduction)
        # Print out the mean/max reduction of CNOT count per number of qubits.
        for nqubits, reduction in d.items():
            print(
                "#qubits: {:2d}, mean reduction of CNOT count: {:0.6f}, "
                "max. reduction: {:d}".format(
                    nqubits, np.mean(np.array(reduction).astype(float)), max(reduction)
                )
            )

    # @unittest.skip("temporary skipping this test")
    def test_full_compressor(self):
        """Tests full compressor."""
        print("\nRunning {:s}() ...".format(self.test_full_compressor.__name__))
        print("Here we check that CNOT compressor does not distort")
        print("circuit matrix, but possibly reduces the number of CNOTs.")

        nL = [
            (repeat, n, L)
            for repeat in range(100)
            for n in range(2, 10 + 1)
            for L in np.arange(1, 100)
        ]

        _debug = False
        if _debug:
            results = list()
            for _, n, L in nL:
                results.append(self._full_compressor(n, L))
        else:
            results = Parallel(n_jobs=-1, prefer="processes")(
                delayed(self._full_compressor)(n, L) for _, n, L in nL
            )
        print("")
        sys.stderr.flush()
        sys.stdout.flush()

        # Group all the results by the number of qubits.
        d = OrderedDict()
        for nqubits, reduction in results:
            d.setdefault(nqubits, []).append(reduction)
        # Print out the mean/max reduction of CNOT count per number of qubits.
        print("\nN O T E: negative reduction value means that compressed")
        print("CNOT structure is longer (!) than the original one because")
        print("of shortcomings of compression algorithm(s).\n")
        for nqubits, reduction in d.items():
            print(
                "#qubits: {:2d}, mean reduction of CNOT count: {:0.6f}, "
                "min. reduction: {:d}, max. reduction: {:d}".format(
                    nqubits,
                    np.mean(np.array(reduction).astype(float)),
                    min(reduction),
                    max(reduction),
                )
            )


if __name__ == "__main__":
    np.set_printoptions(precision=6, linewidth=256)
    unittest.main()

    # TODO: temporary:
    # cnots = np.array([[2, 1, 1, 2, 1, 1, 1, 2],
    #                   [1, 2, 2, 1, 2, 2, 2, 1]])
    # nqubits = 3
    # cnots = np.array([[2, 1, 2, 2, 3, 2, 2, 2, 2, 1, 1, 2, 2, 3, 3, 1, 2, 1, 1, 1, 1, 1, 3,
    # 1, 3, 3, 1],
    #                   [1, 2, 3, 1, 1, 3, 1, 3, 3, 2, 3, 1, 1, 2, 2, 2, 3, 3, 2, 3, 2, 3, 1,
    #                   2, 2, 1, 2]])
    # depth = cnots.shape[1]
