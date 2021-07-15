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
# pylint: disable=no-name-in-module, invalid-name

"""Test M3 matvec"""
import numpy as np
import scipy.sparse.linalg as spla

from qiskit import QuantumCircuit, transpile
from qiskit.test import QiskitTestCase
from qiskit.mitigation.mthree import M3Mitigation
from qiskit.mitigation.mthree.matvec import M3MatVec
from qiskit.test.mock import FakeAthens


class TestMatVec(QiskitTestCase):
    """Test matvec"""

    def test_matvec(self):
        """Check that matvec and rmatvec values are returned as expected"""
        backend = FakeAthens()

        qc = QuantumCircuit(5)
        qc.h(2)
        qc.cx(2, 1)
        qc.cx(2, 3)
        qc.cx(1, 0)
        qc.cx(3, 4)
        qc.measure_all()

        trans_qc = transpile(qc, backend)
        raw_counts = backend.run(trans_qc, backend).result().get_counts()
        mit = M3Mitigation(backend)
        mit.tensored_cals_from_system(range(5))

        cals = mit._form_cals(range(5))
        M = M3MatVec(dict(raw_counts), cals, 5)
        L = spla.LinearOperator((M.num_elems, M.num_elems), matvec=M.matvec)

        LT = spla.LinearOperator((M.num_elems, M.num_elems), matvec=M.rmatvec)

        A = mit.reduced_cal_matrix(raw_counts, range(5))[0]
        vec = (-1) ** np.arange(M.num_elems) * np.ones(M.num_elems, dtype=float) / M.num_elems

        v1 = L.dot(vec)
        v2 = A.dot(vec)

        self.assertTrue(np.allclose(v1, v2, atol=1e-14))

        v3 = LT.dot(vec)
        v4 = (A.T).dot(vec)

        self.assertTrue(np.allclose(v3, v4, atol=1e-14))
