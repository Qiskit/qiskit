# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# =============================================================================

""" Test Quantum Gradient Framework """

import unittest
from test import combine

import numpy as np
from ddt import ddt

from qiskit import QuantumCircuit
from qiskit.algorithms.gradients import (
    FiniteDiffSamplerGradient,
    LinCombSamplerGradient,
    ParamShiftSamplerGradient,
    SPSASamplerGradient,
)
from qiskit.circuit import Parameter
from qiskit.circuit.library import EfficientSU2, RealAmplitudes
from qiskit.exceptions import QiskitError
from qiskit.primitives import Sampler
from qiskit.test import QiskitTestCase


@ddt
class TestSamplerGradient(QiskitTestCase):
    """Test Sampler Gradient"""

    @combine(grad=[FiniteDiffSamplerGradient, ParamShiftSamplerGradient, LinCombSamplerGradient])
    def test_gradient_p(self, grad):
        """Test the sampler gradient for p"""
        sampler = Sampler()
        a = Parameter("a")
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.p(a, 0)
        qc.h(0)
        qc.measure_all()
        gradient = grad(sampler)
        param_list = [[np.pi / 4], [0], [np.pi / 2]]
        correct_results = [
            [{0: -0.5 / np.sqrt(2), 1: 0.5 / np.sqrt(2)}],
            [{0: 0, 1: 0}],
            [{0: -0.499999, 1: 0.499999}],
        ]
        for i, param in enumerate(param_list):
            quasi_dists = gradient.evaluate([qc], [param]).quasi_dists[0]
            for j, quasi_dist in enumerate(quasi_dists):
                for k in quasi_dist:
                    self.assertAlmostEqual(quasi_dist[k], correct_results[i][j][k], 3)

    @combine(grad=[FiniteDiffSamplerGradient, ParamShiftSamplerGradient, LinCombSamplerGradient])
    def test_gradient_u(self, grad):
        """Test the sampler gradient for u"""
        sampler = Sampler()
        a = Parameter("a")
        b = Parameter("b")
        c = Parameter("c")
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.u(a, b, c, 0)
        qc.h(0)
        qc.measure_all()
        gradient = grad(sampler)
        param_list = [[np.pi / 4, 0, 0], [np.pi / 4, np.pi / 4, np.pi / 4]]
        correct_results = [
            [{0: -0.5 / np.sqrt(2), 1: 0.5 / np.sqrt(2)}, {0: 0, 1: 0}, {0: 0, 1: 0}],
            [{0: -0.176777, 1: 0.176777}, {0: -0.426777, 1: 0.426777}, {0: -0.426777, 1: 0.426777}],
        ]
        for i, param in enumerate(param_list):
            quasi_dists = gradient.evaluate([qc], [param]).quasi_dists[0]
            for j, quasi_dist in enumerate(quasi_dists):
                for k in quasi_dist:
                    self.assertAlmostEqual(quasi_dist[k], correct_results[i][j][k], 3)

    @combine(grad=[FiniteDiffSamplerGradient, ParamShiftSamplerGradient, LinCombSamplerGradient])
    def test_gradient_efficient_su2(self, grad):
        """Test the sampler gradient for EfficientSU2"""
        sampler = Sampler()
        qc = EfficientSU2(2, reps=1)
        qc.measure_all()
        gradient = grad(sampler)
        param_list = [
            [np.pi / 4 for param in qc.parameters],
            [np.pi / 2 for param in qc.parameters],
        ]
        correct_results = [
            [
                {
                    0: -0.11963834764831836,
                    1: -0.05713834764831845,
                    2: -0.21875000000000003,
                    3: 0.39552669529663675,
                },
                {
                    0: -0.32230339059327373,
                    1: -0.031250000000000014,
                    2: 0.2339150429449554,
                    3: 0.11963834764831843,
                },
                {
                    0: 0.012944173824159189,
                    1: -0.01294417382415923,
                    2: 0.07544417382415919,
                    3: -0.07544417382415919,
                },
                {
                    0: 0.2080266952966367,
                    1: -0.03125000000000002,
                    2: -0.11963834764831842,
                    3: -0.057138347648318405,
                },
                {
                    0: -0.11963834764831838,
                    1: 0.11963834764831838,
                    2: -0.21875000000000003,
                    3: 0.21875,
                },
                {
                    0: -0.2781092167691146,
                    1: -0.0754441738241592,
                    2: 0.27810921676911443,
                    3: 0.07544417382415924,
                },
                {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
                {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
            ],
            [
                {
                    0: -4.163336342344337e-17,
                    1: 2.7755575615628914e-17,
                    2: -4.163336342344337e-17,
                    3: 0.0,
                },
                {0: 0.0, 1: -1.3877787807814457e-17, 2: 4.163336342344337e-17, 3: 0.0},
                {
                    0: -0.24999999999999994,
                    1: 0.24999999999999994,
                    2: 0.24999999999999994,
                    3: -0.24999999999999994,
                },
                {
                    0: 0.24999999999999994,
                    1: 0.24999999999999994,
                    2: -0.24999999999999994,
                    3: -0.24999999999999994,
                },
                {
                    0: -4.163336342344337e-17,
                    1: 4.163336342344337e-17,
                    2: -4.163336342344337e-17,
                    3: 5.551115123125783e-17,
                },
                {
                    0: -0.24999999999999994,
                    1: 0.24999999999999994,
                    2: 0.24999999999999994,
                    3: -0.24999999999999994,
                },
                {0: 0.0, 1: 2.7755575615628914e-17, 2: 0.0, 3: 2.7755575615628914e-17},
                {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
            ],
        ]
        for i, param in enumerate(param_list):
            quasi_dists = gradient.evaluate([qc], [param]).quasi_dists[0]
            for j, quasi_dist in enumerate(quasi_dists):
                for k in quasi_dist:
                    self.assertAlmostEqual(quasi_dist[k], correct_results[i][j][k], 3)

    @combine(grad=[FiniteDiffSamplerGradient, ParamShiftSamplerGradient, LinCombSamplerGradient])
    def test_gradient_rxx(self, grad):
        """Test the sampler gradient for rxx"""
        sampler = Sampler()
        a = Parameter("a")
        qc = QuantumCircuit(2)
        qc.rxx(a, 0, 1)
        qc.measure_all()
        gradient = grad(sampler)
        param_list = [[np.pi / 4], [np.pi / 2]]
        correct_results = [
            [{0: -0.5 / np.sqrt(2), 1: 0, 2: 0, 3: 0.5 / np.sqrt(2)}],
            [{0: -0.5, 1: 0, 2: 0, 3: 0.5}],
        ]
        for i, param in enumerate(param_list):
            quasi_dists = gradient.evaluate([qc], [param]).quasi_dists[0]
            for j, quasi_dist in enumerate(quasi_dists):
                for k in quasi_dist:
                    self.assertAlmostEqual(quasi_dist[k], correct_results[i][j][k], 3)

    @combine(grad=[FiniteDiffSamplerGradient, ParamShiftSamplerGradient, LinCombSamplerGradient])
    def test_gradient_ryy(self, grad):
        """Test the sampler gradient for ryy"""
        sampler = Sampler()
        a = Parameter("a")
        qc = QuantumCircuit(2)
        qc.ryy(a, 0, 1)
        qc.measure_all()
        gradient = grad(sampler)
        param_list = [[np.pi / 4], [np.pi / 2]]
        correct_results = [
            [{0: -0.5 / np.sqrt(2), 1: 0, 2: 0, 3: 0.5 / np.sqrt(2)}],
            [{0: -0.5, 1: 0, 2: 0, 3: 0.5}],
        ]
        for i, param in enumerate(param_list):
            quasi_dists = gradient.evaluate([qc], [param]).quasi_dists[0]
            for j, quasi_dist in enumerate(quasi_dists):
                for k in quasi_dist:
                    self.assertAlmostEqual(quasi_dist[k], correct_results[i][j][k], 3)

    @combine(grad=[FiniteDiffSamplerGradient, ParamShiftSamplerGradient, LinCombSamplerGradient])
    def test_gradient_rzz(self, grad):
        """Test the sampler gradient for rzz"""
        sampler = Sampler()
        a = Parameter("a")
        qc = QuantumCircuit(2)
        qc.h([0, 1])
        qc.rzz(a, 0, 1)
        qc.h([0, 1])
        qc.measure_all()
        gradient = grad(sampler)
        param_list = [[np.pi / 4], [np.pi / 2]]
        correct_results = [
            [{0: -0.5 / np.sqrt(2), 1: 0, 2: 0, 3: 0.5 / np.sqrt(2)}],
            [{0: -0.5, 1: 0, 2: 0, 3: 0.5}],
        ]
        for i, param in enumerate(param_list):
            quasi_dists = gradient.evaluate([qc], [param]).quasi_dists[0]
            for j, quasi_dist in enumerate(quasi_dists):
                for k in quasi_dist:
                    self.assertAlmostEqual(quasi_dist[k], correct_results[i][j][k], 3)

    @combine(grad=[FiniteDiffSamplerGradient, ParamShiftSamplerGradient, LinCombSamplerGradient])
    def test_gradient_rzx(self, grad):
        """Test the sampler gradient for rzx"""
        sampler = Sampler()
        a = Parameter("a")
        qc = QuantumCircuit(2)
        qc.rzx(a, 0, 1)
        qc.measure_all()
        gradient = grad(sampler)
        param_list = [[np.pi / 4], [np.pi / 2]]
        correct_results = [
            [{0: -0.5 / np.sqrt(2), 1: 0, 2: 0.5 / np.sqrt(2), 3: 0}],
            [{0: -0.5, 1: 0, 2: 0.5, 3: 0}],
        ]
        for i, param in enumerate(param_list):
            quasi_dists = gradient.evaluate([qc], [param]).quasi_dists[0]
            for j, quasi_dist in enumerate(quasi_dists):
                for k in quasi_dist:
                    self.assertAlmostEqual(quasi_dist[k], correct_results[i][j][k], 3)

    @combine(grad=[FiniteDiffSamplerGradient, ParamShiftSamplerGradient, LinCombSamplerGradient])
    def test_gradient_parameter_coefficient(self, grad):
        """Test the sampler gradient for parameter variables with coefficients"""
        sampler = Sampler()
        qc = RealAmplitudes(num_qubits=2, reps=1)
        qc.rz(qc.parameters[0].exp() + 2 * qc.parameters[1], 0)
        qc.rx(3.0 * qc.parameters[0] + qc.parameters[1].sin(), 1)
        qc.u(qc.parameters[0], qc.parameters[1], qc.parameters[3], 1)
        qc.p(2 * qc.parameters[0] + 1, 0)
        qc.rxx(qc.parameters[0] + 2, 0, 1)
        qc.measure_all()
        gradient = grad(sampler)
        param_list = [[np.pi / 4 for _ in qc.parameters], [np.pi / 2 for _ in qc.parameters]]
        correct_results = [
            [
                {
                    0: 0.30014831912265927,
                    1: -0.6634809704357856,
                    2: 0.343589357193753,
                    3: 0.019743294119373426,
                },
                {
                    0: 0.16470607453981906,
                    1: -0.40996282450610577,
                    2: 0.08791803062881773,
                    3: 0.15733871933746948,
                },
                {
                    0: 0.27036068339663866,
                    1: -0.273790986018701,
                    2: 0.12752010079553433,
                    3: -0.12408979817347202,
                },
                {
                    0: -0.2098616294167757,
                    1: -0.2515823946449894,
                    2: 0.21929102305386305,
                    3: 0.24215300100790207,
                },
            ],
            [
                {
                    0: -1.844810060881004,
                    1: 0.04620532700836027,
                    2: 1.6367366426074323,
                    3: 0.16186809126521057,
                },
                {
                    0: 0.07296073407769421,
                    1: -0.021774869186331716,
                    2: 0.02177486918633173,
                    3: -0.07296073407769456,
                },
                {
                    0: -0.07794369186049102,
                    1: -0.07794369186049122,
                    2: 0.07794369186049117,
                    3: 0.07794369186049112,
                },
                {
                    0: 0.0,
                    1: 0.0,
                    2: 0.0,
                    3: 0.0,
                },
            ],
        ]

        for i, param in enumerate(param_list):
            quasi_dists = gradient.evaluate([qc], [param]).quasi_dists[0]
            for j, quasi_dist in enumerate(quasi_dists):
                for k in quasi_dist:
                    self.assertAlmostEqual(quasi_dist[k], correct_results[i][j][k], 3)

    @combine(grad=[FiniteDiffSamplerGradient, ParamShiftSamplerGradient, LinCombSamplerGradient])
    def test_gradient_parameters(self, grad):
        """Test the sampler gradient for parameters"""
        sampler = Sampler()
        a = Parameter("a")
        b = Parameter("b")
        qc = QuantumCircuit(1)
        qc.rx(a, 0)
        qc.rz(b, 0)
        qc.measure_all()
        gradient = grad(sampler)
        param_list = [[np.pi / 4, np.pi / 2]]
        correct_results = [
            [{0: -0.5 / np.sqrt(2), 1: 0.5 / np.sqrt(2)}, {}],
        ]
        for i, param in enumerate(param_list):
            quasi_dists = gradient.evaluate([qc], [param], parameters=[[a]]).quasi_dists[0]
            for j, quasi_dist in enumerate(quasi_dists):
                if correct_results[i][j]:
                    for k in quasi_dist:
                        self.assertAlmostEqual(quasi_dist[k], correct_results[i][j][k], 3)
                else:
                    self.assertEqual(quasi_dist, {})

    @combine(grad=[FiniteDiffSamplerGradient, ParamShiftSamplerGradient, LinCombSamplerGradient])
    def test_gradient_multi_arguments(self, grad):
        """Test the sampler gradient for multiple arguments"""
        sampler = Sampler()
        a = Parameter("a")
        b = Parameter("b")
        qc = QuantumCircuit(1)
        qc.rx(a, 0)
        qc.measure_all()
        qc2 = QuantumCircuit(1)
        qc2.rx(b, 0)
        qc2.measure_all()
        gradient = grad(sampler)
        param_list = [[np.pi / 4], [np.pi / 2]]
        correct_results = [
            [{0: -0.5 / np.sqrt(2), 1: 0.5 / np.sqrt(2)}],
            [{0: -0.499999, 1: 0.499999}],
        ]
        quasi_dists = gradient.evaluate([qc, qc2], param_list).quasi_dists
        for j, q_dists in enumerate(quasi_dists):
            quasi_dist = q_dists[0]
            for k in quasi_dist:
                self.assertAlmostEqual(quasi_dist[k], correct_results[j][0][k], 3)

        c = Parameter("c")
        qc3 = QuantumCircuit(1)
        qc3.rx(c, 0)
        qc3.ry(a, 0)
        qc3.measure_all()
        param_list2 = [[np.pi / 4], [np.pi / 4, np.pi / 4], [np.pi / 4, np.pi / 4]]
        quasi_dists = gradient.evaluate(
            [qc, qc3, qc3], param_list2, parameters=[[a], [c], None]
        ).quasi_dists
        correct_results = [
            [{0: -0.5 / np.sqrt(2), 1: 0.5 / np.sqrt(2)}],
            [{0: -0.25, 1: 0.25} if p == c else {} for p in qc3.parameters],
            [{0: -0.25, 1: 0.25}, {0: -0.25, 1: 0.25}],
        ]
        for i, result in enumerate(quasi_dists):
            for j, q_dists in enumerate(result):
                if correct_results[i][j]:
                    for k in q_dists:
                        self.assertAlmostEqual(q_dists[k], correct_results[i][j][k], 3)
                else:
                    self.assertEqual(q_dists, {})

    @combine(grad=[FiniteDiffSamplerGradient, ParamShiftSamplerGradient, LinCombSamplerGradient])
    def test_gradient_validation(self, grad):
        """Test sampler gradient's validation"""
        sampler = Sampler()
        a = Parameter("a")
        qc = QuantumCircuit(1)
        qc.rx(a, 0)
        qc.measure_all()
        gradient = grad(sampler)
        param_list = [[np.pi / 4], [np.pi / 2]]
        with self.assertRaises(QiskitError):
            gradient.evaluate([qc], param_list)
        with self.assertRaises(QiskitError):
            gradient.evaluate([qc, qc], param_list, parameters=[[a]])
        with self.assertRaises(QiskitError):
            gradient.evaluate([qc], [[np.pi / 4, np.pi / 4]])

    def test_spsa_gradient(self):
        """Test the SPSA sampler gradient"""
        sampler = Sampler()
        a = Parameter("a")
        b = Parameter("b")
        qc = QuantumCircuit(2)
        qc.rx(b, 0)
        qc.rx(a, 1)
        qc.measure_all()
        param_list = [[1, 2]]
        correct_results = [
            [
                {0: 0.2273244, 1: -0.6480598, 2: 0.2273244, 3: 0.1934111},
                {0: -0.2273244, 1: 0.6480598, 2: -0.2273244, 3: -0.1934111},
            ],
        ]
        gradient = SPSASamplerGradient(sampler, seed=123)
        # quasi_dists = gradient.evaluate([qc], param_list).quasi_dists
        for i, param in enumerate(param_list):
            quasi_dists = gradient.evaluate([qc], [param]).quasi_dists[0]
            for j, quasi_dist in enumerate(quasi_dists):
                for k in quasi_dist:
                    self.assertAlmostEqual(quasi_dist[k], correct_results[i][j][k], 3)
        # multi parameters
        param_list2 = [[1, 2], [1, 2], [3, 4]]
        correct_results2 = [
            [
                {0: 0.2273244, 1: -0.6480598, 2: 0.2273244, 3: 0.1934111},
                {0: -0.2273244, 1: 0.6480598, 2: -0.2273244, 3: -0.1934111},
            ],
            [
                {0: -0.2273244, 1: 0.6480598, 2: -0.2273244, 3: -0.1934111} if p == b else {}
                for p in qc.parameters
            ],
            [
                {0: -0.0141129, 1: -0.0564471, 2: -0.3642884, 3: 0.4348484},
                {0: 0.0141129, 1: 0.0564471, 2: 0.3642884, 3: -0.4348484},
            ],
        ]
        gradient = SPSASamplerGradient(sampler, seed=123)
        quasi_dists = gradient.evaluate(
            [qc] * 3, param_list2, parameters=[None, [b], None]
        ).quasi_dists

        for i, result in enumerate(quasi_dists):
            for j, q_dists in enumerate(result):
                if correct_results2[i][j]:
                    for k in q_dists:
                        self.assertAlmostEqual(q_dists[k], correct_results2[i][j][k], 3)
                else:
                    self.assertEqual(q_dists, {})


if __name__ == "__main__":
    unittest.main()
