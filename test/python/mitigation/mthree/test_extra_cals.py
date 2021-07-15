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
# pylint: disable=c-extension-no-member

"""Test M3 extra cals"""
import unittest
import numpy as np
import orjson

from qiskit import QuantumCircuit, transpile
from qiskit.test import QiskitTestCase
from qiskit.mitigation.mthree import M3Mitigation
from qiskit.test.mock import FakeAthens


class TestM3ExtraCals(QiskitTestCase):
    """Test methods"""

    def test_missing_qubit_cal(self):
        """Test if missing calibration is retrived at apply_correcton."""

        qc = QuantumCircuit(5)
        qc.h(2)
        qc.cx(2, 1)
        qc.cx(1, 0)
        qc.cx(2, 3)
        qc.cx(3, 4)
        qc.measure_all()

        backend = FakeAthens()
        trans_qc = transpile(qc, backend)
        raw_counts = backend.run(trans_qc, shots=2048).result().get_counts()

        mit = M3Mitigation(backend)
        mit.tensored_cals_from_system(range(4))

        self.assertTrue(mit.single_qubit_cals[4] is None)

        _ = mit.apply_correction(raw_counts, range(5))

        self.assertTrue(not any(mit.single_qubit_cals[kk] is None for kk in range(5)))

    def test_missing_all_cals(self):
        """Test if calibrations get added if none set before."""

        qc = QuantumCircuit(5)
        qc.h(2)
        qc.cx(2, 1)
        qc.cx(1, 0)
        qc.cx(2, 3)
        qc.cx(3, 4)
        qc.measure_all()

        backend = FakeAthens()
        trans_qc = transpile(qc, backend)
        raw_counts = backend.run(trans_qc, shots=2048).result().get_counts()

        mit = M3Mitigation(backend)
        _ = mit.apply_correction(raw_counts, range(5))

        assert not any(mit.single_qubit_cals[kk] is None for kk in range(5))

    @unittest.skip("Need to figure out path.")
    def test_save_cals(self, tmp_path):
        """Test if passing a calibration file saves the correct JSON."""
        backend = FakeAthens()
        cal_file = tmp_path / "cal.json"
        mit = M3Mitigation(backend)
        mit.tensored_cals_from_system(counts_file=cal_file)
        with open(cal_file, "r") as fd:
            cals = np.array(orjson.loads(fd.read()))
        self.assertTrue(np.array_equal(mit.single_qubit_cals, cals))

    @unittest.skip("Need to figure out path.")
    def test_load_cals(self, tmp_path):
        """Test if loading a calibration JSON file correctly loads the cals."""
        cal_file = tmp_path / "cal.json"
        backend = FakeAthens()
        mit = M3Mitigation(backend)
        mit.tensored_cals_from_system(counts_file=cal_file)
        new_mit = M3Mitigation(backend)
        new_mit.tensored_cals_from_file(cal_file)
        self.assertTrue(np.array_equal(mit.single_qubit_cals, new_mit.single_qubit_cals))
