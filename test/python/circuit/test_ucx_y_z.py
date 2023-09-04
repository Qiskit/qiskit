# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for uniformly controlled Rx,Ry and Rz gates"""

import itertools
import unittest
from ddt import ddt, data

import numpy as np
from scipy.linalg import block_diag

from qiskit import BasicAer, QuantumCircuit, QuantumRegister, execute
from qiskit.test import QiskitTestCase

from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.compiler import transpile
from qiskit.circuit.library import UCRXGate, UCRYGate, UCRZGate

angles_list = [
    [0],
    [0.4],
    [0, 0],
    [0, 0.8],
    [0, 0, 1, 1],
    [0, 1, 0.5, 1],
    (2 * np.pi * np.random.rand(2**3)).tolist(),
    (2 * np.pi * np.random.rand(2**4)).tolist(),
    (2 * np.pi * np.random.rand(2**5)).tolist(),
]

rot_axis_list = ["X", "Y", "Z"]


@ddt
class TestUCRXYZ(QiskitTestCase):
    """Qiskit tests for UCRXGate, UCRYGate and UCRZGate rotations gates."""

    @data(True, False)
    def test_ucy(self, use_method):
        """Test the decomposition of uniformly controlled rotations."""
        methods = {"X": "ucrx", "Y": "ucry", "Z": "ucrz"}
        gates = {"X": UCRXGate, "Y": UCRYGate, "Z": UCRZGate}

        for angles, rot_axis in itertools.product(angles_list, rot_axis_list):
            with self.subTest(angles=angles, rot_axis=rot_axis):
                num_contr = int(np.log2(len(angles)))
                q = QuantumRegister(num_contr + 1)
                qc = QuantumCircuit(q)
                if use_method:
                    with self.assertWarns(PendingDeprecationWarning):
                        getattr(qc, methods[rot_axis])(angles, q[1 : num_contr + 1], q[0])
                else:
                    gate = gates[rot_axis](angles)
                    qc.append(gate, q)

                # Decompose the gate
                qc = transpile(qc, basis_gates=["u1", "u3", "u2", "cx", "id"])
                # Simulate the decomposed gate
                simulator = BasicAer.get_backend("unitary_simulator")
                result = execute(qc, simulator).result()
                unitary = result.get_unitary(qc)
                unitary_desired = _get_ucr_matrix(angles, rot_axis)
                self.assertTrue(matrix_equal(unitary_desired, unitary, ignore_phase=True))


def _get_ucr_matrix(angles, rot_axis):
    if rot_axis == "X":
        gates = [
            np.array(
                [
                    [np.cos(angle / 2), -1j * np.sin(angle / 2)],
                    [-1j * np.sin(angle / 2), np.cos(angle / 2)],
                ]
            )
            for angle in angles
        ]
    elif rot_axis == "Y":
        gates = [
            np.array(
                [[np.cos(angle / 2), -np.sin(angle / 2)], [np.sin(angle / 2), np.cos(angle / 2)]]
            )
            for angle in angles
        ]
    else:
        gates = [
            np.array([[np.exp(-1.0j * angle / 2), 0], [0, np.exp(1.0j * angle / 2)]])
            for angle in angles
        ]
    return block_diag(*gates)


if __name__ == "__main__":
    unittest.main()
