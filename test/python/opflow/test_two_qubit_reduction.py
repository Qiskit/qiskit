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

""" Test TwoQubitReduction """

from test.python.opflow import QiskitOpflowTestCase

from qiskit.opflow import PauliSumOp, TwoQubitReduction, TaperedPauliSumOp, Z2Symmetries
from qiskit.quantum_info import Pauli, SparsePauliOp


class TestTwoQubitReduction(QiskitOpflowTestCase):
    """TwoQubitReduction tests."""

    def test_convert(self):
        """convert test"""

        qubit_op = PauliSumOp.from_list(
            [
                ("IIII", -0.8105479805373266),
                ("IIIZ", 0.17218393261915552),
                ("IIZZ", -0.22575349222402472),
                ("IZZI", 0.1721839326191556),
                ("ZZII", -0.22575349222402466),
                ("IIZI", 0.1209126326177663),
                ("IZZZ", 0.16892753870087912),
                ("IXZX", -0.045232799946057854),
                ("ZXIX", 0.045232799946057854),
                ("IXIX", 0.045232799946057854),
                ("ZXZX", -0.045232799946057854),
                ("ZZIZ", 0.16614543256382414),
                ("IZIZ", 0.16614543256382414),
                ("ZZZZ", 0.17464343068300453),
                ("ZIZI", 0.1209126326177663),
            ]
        )
        tapered_qubit_op = TwoQubitReduction(num_particles=2).convert(qubit_op)
        self.assertIsInstance(tapered_qubit_op, TaperedPauliSumOp)

        primitive = SparsePauliOp.from_list(
            [
                ("II", -1.052373245772859),
                ("ZI", -0.39793742484318007),
                ("IZ", 0.39793742484318007),
                ("ZZ", -0.01128010425623538),
                ("XX", 0.18093119978423142),
            ]
        )
        symmetries = [Pauli("IIZI"), Pauli("ZIII")]
        sq_paulis = [Pauli("IIXI"), Pauli("XIII")]
        sq_list = [1, 3]
        tapering_values = [-1, 1]
        z2_symmetries = Z2Symmetries(symmetries, sq_paulis, sq_list, tapering_values)
        expected_op = TaperedPauliSumOp(primitive, z2_symmetries)
        self.assertEqual(tapered_qubit_op, expected_op)
