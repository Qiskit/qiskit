# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the quantum linear system solver algorithm."""

import unittest
from test.python.algorithms import QiskitAlgorithmsTestCase
from scipy.linalg import expm
from scipy.sparse import diags
import numpy as np
from ddt import ddt, idata, data, unpack
from qiskit import QuantumRegister, QuantumCircuit, BasicAer
from qiskit.algorithms.linear_solvers.hhl import HHL
from qiskit.algorithms.linear_solvers.matrices.tridiagonal import Tridiagonal
from qiskit.algorithms.linear_solvers.observables.absolute_average import AbsoluteAverage
from qiskit.algorithms.linear_solvers.observables.matrix_functional import MatrixFunctional
from qiskit.quantum_info import Operator, Statevector, partial_trace
from qiskit.opflow import I, Z


@ddt
class TestTridiagonal(QiskitAlgorithmsTestCase):
    """Tests based on the Tridiagonal class.

    This class tests
        * the constructed circuits
    """
    @idata([
        [Tridiagonal(2, 1, -1/3)],
        [Tridiagonal(3, 2, 1), 1.1, 3]
    ])
    @unpack
    def test_statevector(self, matrix, time=1.0, power=1):
        """ statevector test """
        if time is not None:
            matrix.evo_time = time

        num_qubits = matrix.num_state_qubits
        pow_circ = matrix.power(power).control()
        circ_qubits = pow_circ.num_qubits
        qc = QuantumCircuit(circ_qubits)
        qc.append(matrix.power(power).control(), list(range(circ_qubits)))
        circ_matrix = Operator(qc).data
        # extract the parts of the circuit matrix corresponding to Tridiagonal
        ZeroOp = ((I + Z) / 2)
        OneOp = ((I - Z) / 2)
        proj = Operator((ZeroOp ^ pow_circ.num_ancillas) ^ (I ^ num_qubits) ^ OneOp).data
        circ_matrix = Operator(qc).data
        approx_exp = partial_trace(np.dot(proj, circ_matrix), [0] +
                                   list(range(num_qubits + 1, circ_qubits))).data

        exact_exp = expm(1j * matrix.evo_time * power * matrix.matrix)
        np.testing.assert_array_almost_equal(approx_exp, exact_exp, decimal=2)


if __name__ == '__main__':
    unittest.main()
