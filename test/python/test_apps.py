# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""Quick program to test the apps tools modules."""

import unittest
from unittest.mock import patch
from functools import partial
from io import StringIO
from scipy import linalg as la
import numpy as np

from qiskit.tools.apps.optimization import make_Hamiltonian, Hamiltonian_from_file, group_paulis, \
    eval_hamiltonian, trial_circuit_ry, trial_circuit_ryrz, SPSA_calibration, SPSA_optimization, \
    print_pauli_list_grouped, Energy_Estimate
from qiskit.tools.apps.fermion import parity_set, update_set, flip_set, fermionic_maps, \
    two_qubit_reduction
from qiskit.tools.qi.pauli import Pauli
from qiskit import QuantumProgram

from .common import QiskitTestCase


class TestQuantumOptimization(QiskitTestCase):
    """Tests for quantum optimization"""

    def test_trial_functions(self):
        """trial_circuit_ry function"""
        entangler_map = {0: [2], 1: [2], 3: [2], 4: [2]}

        m = 1
        n = 6
        theta = np.zeros(m * n)

        trial_circuit = trial_circuit_ry(n, m, theta, entangler_map)
        qasm_txt = trial_circuit.qasm()
        self.log.info(qasm_txt)
        self.assertEqual(len(qasm_txt), 456)

        self.log.info("With No measurement:\n")
        trial_circuit = trial_circuit_ry(n, m, theta, entangler_map, None, None)
        qasm_txt = trial_circuit.qasm()
        self.log.info(qasm_txt)
        self.assertEqual(len(qasm_txt), 324)

        self.log.info("With Y measurement:\n")
        meas_sting = ['Y' for x in range(n)]
        trial_circuit = trial_circuit_ry(n, m, theta, entangler_map, meas_sting)
        qasm_txt = trial_circuit.qasm()
        self.log.info(qasm_txt)
        self.assertEqual(len(qasm_txt), 564)

    def test_optimization_of_H2_at_bond_length(self):
        """From chemistry tutorial, but shorter.

        https://github.com/QISKit/qiskit-tutorial/blob/master/\
        4_applications/quantum_chemistry.ipynb#Optimization-of-H2-at-bond-length but shorter."""
        n = 2
        m = 6
        device = 'local_statevector_simulator'

        np.random.seed(42)
        initial_theta = np.random.randn(2 * n * m)
        entangler_map = {
            0: [1]}  # the map of two-qubit gates with control at key and target at values
        shots = 1
        max_trials = 1
        ham_name = self._get_resource_path("../performance/H2/H2Equilibrium.txt")

        # Exact Energy
        pauli_list = Hamiltonian_from_file(ham_name)
        H = make_Hamiltonian(pauli_list)
        exact = np.amin(la.eig(H)[0]).real
        self.log.info('The exact ground state energy is: %s', exact)
        self.assertEqual(exact, -1.8572746704950902)

        # Optimization
        Q_program = QuantumProgram()

        def cost_function(Q_program, H, n, m, entangler_map, shots, device, theta):
            # pylint: disable=missing-docstring
            return eval_hamiltonian(Q_program, H,
                                    trial_circuit_ryrz(n, m, theta, entangler_map, None, False),
                                    shots, device).real

        initial_c = 0.01
        target_update = 2 * np.pi * 0.1

        expected_stout = ("calibration step # 0 of 1\n"
                          "calibrated SPSA_parameters[0] is 2.5459894")
        with patch('sys.stdout', new=StringIO()) as fakeOutput:
            SPSA_params = SPSA_calibration(partial(cost_function, Q_program, H, n, m, entangler_map,
                                                   shots, device), initial_theta, initial_c,
                                           target_update, 1)
        self.assertMultiLineEqual(fakeOutput.getvalue().strip(), expected_stout)

        expected_stout = ("objective function at theta+ for step # 0\n"
                          "-1.0909948\n"
                          "objective function at theta- for step # 0\n"
                          "-1.0675805\n"
                          "Final objective function is: -1.2619548")
        with patch('sys.stdout', new=StringIO()) as fakeOutput:
            output = SPSA_optimization(
                partial(cost_function, Q_program, H, n, m, entangler_map, shots, device),
                initial_theta, SPSA_params, max_trials)

        self.assertMultiLineEqual(fakeOutput.getvalue().strip(), expected_stout)

        output1 = np.array([-2.48391736629, 2.84236721813, -2.3329429812, 4.50366137571,
                            -3.21478489403, -3.21476847625, 4.55984433481, -2.21319679015,
                            2.51115713337, 3.52319156289, 2.51721382649, 2.51490176573,
                            3.22259379087, 1.06735127465, 1.25571368679, 2.41834399006,
                            1.96780039897, 3.2948788519, 2.07260744378, -4.39293522064,
                            -1.51498275038, 2.75485521882, 3.04815972399, 1.55588333309])
        output4 = np.array([0.486714153011, -0.128264301171, 0.637688538101, 1.53302985641,
                            -0.244153374723, -0.244136956949, 1.58921281551, 0.757434729153,
                            -0.459474385935, 0.552560043586, -0.453417692812, -0.45572975357,
                            0.251962271566, -1.90328024466, -1.71491783251, -0.552287529241,
                            -1.00283112033, 0.324247332595, -0.898024075521, -1.42230370134,
                            1.45564876892, -0.215776300487, 0.0775282046879, -1.41474818621])
        output5 = np.array([0.506714153011, -0.148264301171, 0.657688538101, 1.51302985641,
                            -0.224153374723, -0.224136956949, 1.56921281551, 0.777434729153,
                            -0.479474385935, 0.532560043586, -0.473417692812, -0.47572975357,
                            0.231962271566, -1.92328024466, -1.73491783251, -0.572287529241,
                            -1.02283112033, 0.304247332595, -0.918024075521, -1.40230370134,
                            1.47564876892, -0.235776300487, 0.0575282046879, -1.43474818621])

        self.assertEqual(6, len(output))
        np.testing.assert_almost_equal(-1.2619547992193472, output[0])
        self.assertEqual(output1.all(), output[1].all())
        np.testing.assert_almost_equal([-1.0909948471209499], output[2])
        np.testing.assert_almost_equal([-1.0675805189515357], output[3])
        self.assertEqual(1, len(output[4]))
        self.assertEqual(output4.all(), output[4][0].all())
        self.assertEqual(output5.all(), output[5][0].all())

    def test_group_paulis(self):
        """ qiskit.tools.apps.optimization.group_paulis function"""
        ham_name = self._get_resource_path("../performance/H2/H2Equilibrium.txt")
        zz = np.array([0, 0])
        oo = np.array([1, 1])

        pauli_list = Hamiltonian_from_file(ham_name)
        pauli_list_grouped = group_paulis(pauli_list)

        self.assertEqual(len(pauli_list_grouped), 2)
        r0 = [i[0] for i in pauli_list_grouped]
        r1 = [i[1] for i in pauli_list_grouped]

        self.assertEqual(len(r0), 2)
        r00 = [i[0] for i in r0]
        r01 = [i[1] for i in r0]
        e01 = [Pauli(oo, zz), Pauli(zz, oo)]
        self.assertEqual([0, 0], r00)
        self.assertEqual(e01, r01)

        self.assertEqual(len(r1), 2)
        r10 = [i[0] for i in r1]
        r11 = [i[1] for i in r1]
        e11 = [Pauli(oo, zz), Pauli(zz, oo)]
        self.assertEqual([0.011279956224107712, 0.18093133934472627], r10)
        self.assertEqual(e11, r11)

        expected_stout = ("Post Rotations of TPB set 0:\nZZ\n0\n\nZZ\n0.0112800\n"
                          "II\n-1.0523761\nZI\n0.3979357\nIZ\n"
                          "0.3979357\n\n\nPost Rotations of TPB set 1:\nXX\n0\n\n"
                          "XX\n0.1809313")
        with patch('sys.stdout', new=StringIO()) as fakeOutput:
            print_pauli_list_grouped(pauli_list_grouped)

        self.assertMultiLineEqual(fakeOutput.getvalue().strip(), expected_stout)

    def test_eval_hamiltonian_pauli_list(self):
        """
        Test of trial_circuit_ry and eval_hamiltonian with a pauli list
        """
        ham_name = self._get_resource_path("../performance/H2/H2Equilibrium.txt")
        pauli_list = Hamiltonian_from_file(ham_name)
        n = 2
        m = 6
        device = 'local_statevector_simulator'

        theta = np.array([-0.607547697211, -0.126136414606, - 0.684606358705, 0.928714748593,
                          -1.84440103405, -0.467002424077, 2.29249034315, 0.488810054396,
                          0.710266990661, 1.05553444322, 0.0540731003458, 0.257953416342,
                          0.588281649703, 0.885244238613, -1.01700702421, -0.133693031283,
                          -0.438185501332, 0.493443494428, -0.199009119848, -1.27498360732,
                          0.293494154438, 0.108950311827, 0.031726785859, 1.27263986303])
        entangler_map = {0: [1]}

        energy = eval_hamiltonian(QuantumProgram(), pauli_list,
                                  trial_circuit_ry(n, m, theta, entangler_map, None, False), 1,
                                  device)
        np.testing.assert_almost_equal(-0.45295043823057191 + 3.3552033732997923e-18j, energy)


class TestHamiltonian(QiskitTestCase):
    """ Energy_Estimate and make_Hamiltonian functions """

    def test_hamiltonian(self):
        # pylint: disable=missing-docstring,unexpected-keyword-arg
        # printing an example from a H2 file
        hfile = self._get_resource_path("../performance/H2/H2Equilibrium.txt")
        hamiltonian = make_Hamiltonian(Hamiltonian_from_file(hfile))
        self.log.info(hamiltonian)
        # [[-0.24522469381221926 0 0 0.18093133934472627 ]
        # [0 -1.0636560168497590 0.18093133934472627 0]
        # [0 0.18093133934472627 -1.0636560168497592 0]
        # [0.18093133934472627 0 0 -1.8369675149908681]]

        expected_result = [
            [(-0.245224693812 + 0j), 0j, 0j, (0.180931339345 + 0j)],
            [0j, (-1.06365601685 + 0j), (0.180931339345 + 0j), 0j],
            [0j, (0.180931339345 + 0j), (-1.06365601685 + 0j), 0j],
            [(0.180931339345 + 0j), 0j, 0j, (-1.83696751499 + 0j)]
        ]

        for i in range(4):
            with self.subTest(i=i):
                for result, expected in zip(hamiltonian[i], expected_result[i]):
                    np.testing.assert_almost_equal(result, expected)

        # printing an example from a graph input
        n = 3
        v0 = np.zeros(n)
        v0[2] = 1
        v1 = np.zeros(n)
        v1[0] = 1
        v1[1] = 1
        v2 = np.zeros(n)
        v2[0] = 1
        v2[2] = 1
        v3 = np.zeros(n)
        v3[1] = 1
        v3[2] = 1

        pauli_list = [(1, Pauli(v0, np.zeros(n))), (1, Pauli(v1, np.zeros(n))),
                      (1, Pauli(v2, np.zeros(n))), (1, Pauli(v3, np.zeros(n)))]
        a = make_Hamiltonian(pauli_list)
        self.log.info(a)

        w, v = la.eigh(a, eigvals=(0, 0))
        self.log.info(w)
        self.log.info(v)

        data = {'000': 10}
        self.log.info(Energy_Estimate(data, pauli_list))
        data = {'001': 10}
        self.log.info(Energy_Estimate(data, pauli_list))
        data = {'010': 10}
        self.log.info(Energy_Estimate(data, pauli_list))
        data = {'011': 10}
        self.log.info(Energy_Estimate(data, pauli_list))
        data = {'100': 10}
        self.log.info(Energy_Estimate(data, pauli_list))
        data = {'101': 10}
        self.log.info(Energy_Estimate(data, pauli_list))
        data = {'110': 10}
        self.log.info(Energy_Estimate(data, pauli_list))
        data = {'111': 10}
        self.log.info(Energy_Estimate(data, pauli_list))


class TestAppsFermion(QiskitTestCase):
    """Tests for apps"""

    def setUp(self):
        self.j1 = 8
        self.j2 = 3
        self.j3 = 5
        self.n = 10

        self.a2 = np.arange(4).reshape(2, 2)
        self.a5 = np.arange(16).reshape(2, 2, 2, 2)
        self.e0 = [np.complex(-0.75, 0),
                   np.complex(0.75, 0),
                   np.complex(0, -0.25),
                   np.complex(0, 0.25),
                   np.complex(-1.5, 0),
                   np.complex(1.5, 0)]
        self.zz = np.array([0, 0])
        self.oz = np.array([1, 0])
        self.zo = np.array([0, 1])
        self.oo = np.array([1, 1])

    def test_parity_set(self):
        """ qiskit.tools.apps.fermion.parity_set function"""
        r = parity_set(self.j1, self.n)
        self.assertEqual(r, [4.])
        r = parity_set(self.j2, self.n)
        self.assertEqual(r.size, 0)

    def test_update_set(self):
        """ qiskit.tools.apps.fermion.update_set function"""
        r = update_set(self.j1, self.n)
        self.assertEqual(r.size, 0)
        r = update_set(self.j2, self.n)
        self.assertEqual(r, [9.])

    def test_flip_set(self):
        """ qiskit.tools.apps.fermion.flip_set function"""
        r = flip_set(self.j1, self.n)
        self.assertEqual(r.size, 0)
        r = flip_set(self.j2, self.n)
        self.assertEqual(r.size, 0)
        r = flip_set(self.j3, self.n)
        self.assertEqual(r.size, 0)

    def test_fermionic_maps_jordan_wigner(self):
        """ qiskit.tools.apps.fermion.fermionic_maps with JORDAN_WIGNER map type"""
        self.e0[0] = np.complex(0.75, 0)
        r = fermionic_maps(self.a2, self.a5, "JORDAN_WIGNER")
        self.assertEqual(len(r), 6)
        r0 = [i[0] for i in r]
        self.assertEqual(self.e0, r0)
        r1 = [i[1] for i in r]
        e = [Pauli(self.oo, self.oo),
             Pauli(self.zz, self.oo),
             Pauli(self.zo, self.oo),
             Pauli(self.oz, self.oo),
             Pauli(self.zo, self.zz),
             Pauli(self.zz, self.zz)]
        self.assertEqual(r1, e)

    def test_fermionic_maps_parity(self):
        """ qiskit.tools.apps.fermion.fermionic_maps with PARITY map type"""
        r = fermionic_maps(self.a2, self.a5, "PARITY")
        self.assertEqual(len(r), 6)
        r0 = [i[0] for i in r]
        self.assertEqual(self.e0, r0)
        r1 = [i[1] for i in r]
        e = [Pauli(self.zo, self.oz),
             Pauli(self.zz, self.oz),
             Pauli(self.oo, self.oz),
             Pauli(self.oz, self.oz),
             Pauli(self.oo, self.zz),
             Pauli(self.zz, self.zz)]
        self.assertEqual(r1, e)

    def test_fermionic_maps_binary_tree(self):
        """ qiskit.tools.apps.fermion.fermionic_maps with BINARY_TREE map type"""
        r = fermionic_maps(self.a2, self.a5, "BINARY_TREE")
        self.assertEqual(len(r), 6)
        r0 = [i[0] for i in r]
        self.assertEqual(self.e0, r0)
        r1 = [i[1] for i in r]
        e = [Pauli(self.zo, self.oz),
             Pauli(self.zz, self.oz),
             Pauli(self.oo, self.oz),
             Pauli(self.oz, self.oz),
             Pauli(self.oo, self.zz),
             Pauli(self.zz, self.zz)]
        self.assertEqual(r1, e)

    def test_two_qubit_reduction(self):
        """ qiskit.tools.apps.fermion.two_qubit_reduction"""
        pauli_list = []
        n = 4
        w = np.arange(n ** 2).reshape(n, n)

        for i in range(n):
            for j in range(i):
                if w[i, j] != 0:
                    wp = np.zeros(n)
                    vp = np.zeros(n)
                    vp[n - i - 1] = 1
                    vp[n - j - 1] = 1
                    pauli_list.append((w[i, j], Pauli(vp, wp)))
        r = two_qubit_reduction(pauli_list, 10)
        r0 = [i[0] for i in r]
        self.assertEqual([-5, -8, -2, 13], r0)
        r1 = [i[1] for i in r]
        e = [Pauli(self.zo, self.zz),
             Pauli(self.zz, self.zz),
             Pauli(self.oz, self.zz),
             Pauli(self.oo, self.zz)]
        self.assertEqual(e, r1)


if __name__ == '__main__':
    unittest.main()
