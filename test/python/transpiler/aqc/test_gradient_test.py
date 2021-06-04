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
Tests analytical gradient vs the one computed via finite differences.
"""
print("\n{:s}\n{:s}\n{:s}\n".format("@" * 80, __doc__, "@" * 80))

import sys, os, traceback

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
import numpy as np
import unittest
from joblib import Parallel, delayed
import qiskit.transpiler.synthesis.aqc.utils as ut
from qiskit.transpiler.synthesis.aqc.parametric_circuit import ParametricCircuit

# Avoid excessive deprecation warnings in Qiskit on Linux system.
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


class TestGradientAgainstFiniteDiff(unittest.TestCase):
    """
    Compares analytical gradient vs the one computed via finite difference
    approximation. Also, the test demonstrates that the difference between
    analytical and numerical gradients is up to quadratic term in Taylor
    expansion for small deltas.
    """

    def _gradient_test(self, nqubits: int, depth: int) -> (int, int, list, list):
        """
        Gradient test for specified number of qubits and circuit depth.
        """
        print(".", end="", flush=True)
        self.assertTrue(isinstance(nqubits, (int, np.int64)))
        self.assertTrue(isinstance(depth, (int, np.int64)))
        _TINY = float(np.power(np.finfo(np.float64).tiny, 0.2))
        circuit = ParametricCircuit(
            num_qubits=nqubits, layout="spin", connectivity="full", depth=depth
        )
        circuit.init_gradient_backend(backend="default")

        # Generate random target matrix and random starting point. Repeat until
        # sufficiently large gradient has been encountered.
        while True:
            target_matrix = ut.random_SU(nqubits=nqubits)
            thetas = np.random.rand(circuit.num_thetas) * (2.0 * np.pi)
            circuit.set_thetas(thetas)
            fobj0, grad0 = circuit.get_gradient(target_matrix=target_matrix)
            if np.linalg.norm(grad0) > 1e-2:
                break

        grad0_dir = grad0 / np.linalg.norm(grad0)
        numerical_grad = np.zeros(thetas.size)
        thetas_delta = np.zeros(thetas.size)

        # Every angle has a magnitude between 0 and 2*pi. We choose the
        # angle increment (tau) about the same order of magnitude, tau <= 1,
        # and then gradually decrease it towards zero.
        tau = 1.0
        diff_prev = 0.0
        orders = list()
        errors = list()
        for step in range(16):
            # Estimate gradient approximation error.
            for i in range(thetas.size):
                np.copyto(thetas_delta, thetas)
                thetas_delta[i] -= tau
                circuit.set_thetas(thetas_delta)
                fobj1, _ = circuit.get_gradient(target_matrix=target_matrix)
                np.copyto(thetas_delta, thetas)
                thetas_delta[i] += tau
                circuit.set_thetas(thetas_delta)
                fobj2, _ = circuit.get_gradient(target_matrix=target_matrix)
                numerical_grad[i] = (fobj2 - fobj1) / (2.0 * tau)
            errors.append(np.linalg.norm(grad0 - numerical_grad) / np.linalg.norm(grad0))

            # Estimate approximation order (should be quadratic for small tau).
            # Note, we take perturbation in gradient direction. More rigorous
            # approach would take a random direction, although quadratic
            # convergence is less pronounced in this case.
            perturbation = grad0_dir * tau
            circuit.set_thetas(thetas + perturbation)
            fobj, _ = circuit.get_gradient(target_matrix=target_matrix)
            diff = abs(fobj - fobj0 - np.dot(grad0, perturbation))
            orders.append(
                0.0 if step == 0 else float((np.log(diff_prev) - np.log(diff)) / np.log(2.0))
            )

            tau /= 2.0
            diff_prev = diff

        # Important when run inside a parallel process:
        sys.stderr.flush()
        sys.stdout.flush()
        return int(nqubits), int(depth), errors, orders

    def test_gradient_test(self):
        print("\nRunning {:s}() ...".format(self.test_gradient_test.__name__))
        print("Here we compare gradient of a circuit against the one computed")
        print("via finite difference. Also, we try to reveal quadratic nature")
        print("of the gradient difference (quadratic term in Taylor expansion).")
        print("Ideally, the accuracy of gradient approximation improves from")
        print("left to right as delta goes down to zero, and the order of")
        print("approximation residual approaches 2 (quadratic Taylor's term).")
        print("\n")

        nL = [(n, L) for n in range(2, 7) for L in np.random.permutation(np.arange(10, 30))[0:10]]

        results = Parallel(n_jobs=-1, prefer="processes")(
            delayed(self._gradient_test)(n, L) for n, L in nL
        )
        print("")
        sys.stderr.flush()
        sys.stdout.flush()

        # Print out the results.
        np.set_printoptions(precision=3, linewidth=256)  # reduced precision!
        for nqubits, depth, errors, orders in results:
            print(
                "#qubits: {:d}, circuit depth: {:d}\n"
                "Accuracy: {}\nOrder: {}\n{:s}".format(
                    nqubits, depth, np.array(errors), np.array(orders), "-" * 80
                )
            )


if __name__ == "__main__":
    np.set_printoptions(precision=6, linewidth=256)
    try:
        unittest.main()
    except Exception as ex:
        print("message length:", len(str(ex)))
        traceback.print_exc()
