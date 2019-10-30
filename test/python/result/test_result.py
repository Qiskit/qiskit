# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test Qiskit's Result class."""

import numpy as np

from qiskit.result import models
from qiskit.validation import base
from qiskit.result import Result
from qiskit.test import QiskitTestCase


class TestResultOperations(QiskitTestCase):
    """Result operations methods."""

    def setUp(self):
        self.base_result_args = dict(backend_name='test_backend',
                                     backend_version='1.0.0',
                                     qobj_id='id-123',
                                     job_id='job-123',
                                     success=True)

        super().setUp()

    def test_counts_no_header(self):
        """Test that counts are extracted properly without header."""
        raw_counts = {'0x0': 4, '0x2': 10}
        no_header_processed_counts = {bin(int(bs[2:], 16))[2:]: counts for
                                      (bs, counts) in raw_counts.items()}
        data = models.ExperimentResultData(counts=base.Obj(**raw_counts))
        exp_result = models.ExperimentResult(shots=14, success=True, meas_level=2, data=data)
        result = Result(results=[exp_result], **self.base_result_args)

        self.assertEqual(result.get_counts(0), no_header_processed_counts)

    def test_counts_header(self):
        """Test that counts are extracted properly with header."""
        raw_counts = {'0x0': 4, '0x2': 10}
        processed_counts = {'0 0 00': 4, '0 0 10': 10}
        data = models.ExperimentResultData(counts=base.Obj(**raw_counts))
        exp_result_header = base.Obj(creg_sizes=[['c0', 2], ['c0', 1], ['c1', 1]],
                                     memory_slots=4)
        exp_result = models.ExperimentResult(shots=14, success=True, meas_level=2,
                                             data=data, header=exp_result_header)
        result = Result(results=[exp_result], **self.base_result_args)

        self.assertEqual(result.get_counts(0), processed_counts)

    def test_memory_counts_no_header(self):
        """Test that memory bitstrings are extracted properly without header."""
        raw_memory = ['0x0', '0x0', '0x2', '0x2', '0x2', '0x2', '0x2']
        no_header_processed_memory = [bin(int(bs[2:], 16))[2:] for bs in raw_memory]
        data = models.ExperimentResultData(memory=raw_memory)
        exp_result = models.ExperimentResult(shots=14, success=True, meas_level=2,
                                             memory=True, data=data)
        result = Result(results=[exp_result], **self.base_result_args)

        self.assertEqual(result.get_memory(0), no_header_processed_memory)

    def test_memory_counts_header(self):
        """Test that memory bitstrings are extracted properly with header."""
        raw_memory = ['0x0', '0x0', '0x2', '0x2', '0x2', '0x2', '0x2']
        no_header_processed_memory = ['0 0 00', '0 0 00', '0 0 10', '0 0 10',
                                      '0 0 10', '0 0 10', '0 0 10']
        data = models.ExperimentResultData(memory=raw_memory)
        exp_result_header = base.Obj(creg_sizes=[['c0', 2], ['c0', 1], ['c1', 1]],
                                     memory_slots=4)
        exp_result = models.ExperimentResult(shots=14, success=True, meas_level=2,
                                             memory=True, data=data,
                                             header=exp_result_header)
        result = Result(results=[exp_result], **self.base_result_args)

        self.assertEqual(result.get_memory(0), no_header_processed_memory)

    def test_meas_level_1_avg(self):
        """Test measurement level 1 average result."""
        # 3 qubits
        raw_memory = [[0., 1.], [1., 0.], [0.5, 0.5]]
        processed_memory = np.array([1.j, 1., 0.5+0.5j], dtype=np.complex_)
        data = models.ExperimentResultData(memory=raw_memory)
        exp_result = models.ExperimentResult(shots=2, success=True, meas_level=1,
                                             meas_return='avg', data=data)
        result = Result(results=[exp_result], **self.base_result_args)
        memory = result.get_memory(0)

        self.assertEqual(memory.shape, (3,))
        self.assertEqual(memory.dtype, np.complex_)
        np.testing.assert_almost_equal(memory, processed_memory)

    def test_meas_level_1_single(self):
        """Test measurement level 1 single result."""
        # 3 qubits
        raw_memory = [[[0., 1.], [1., 0.], [0.5, 0.5]],
                      [[0.5, 0.5], [1., 0.], [0., 1.]]]
        processed_memory = np.array([[1.j, 1., 0.5+0.5j],
                                     [0.5+0.5j, 1., 1.j]], dtype=np.complex_)
        data = models.ExperimentResultData(memory=raw_memory)
        exp_result = models.ExperimentResult(shots=2, success=True, meas_level=1,
                                             meas_return='single', data=data)
        result = Result(results=[exp_result], **self.base_result_args)
        memory = result.get_memory(0)

        self.assertEqual(memory.shape, (2, 3))
        self.assertEqual(memory.dtype, np.complex_)
        np.testing.assert_almost_equal(memory, processed_memory)

    def test_meas_level_0_avg(self):
        """Test measurement level 0 average result."""
        # 3 qubits
        raw_memory = [[[0., 1.], [0., 1.], [0., 1.]],
                      [[1., 0.], [1., 0.], [1., 0.]]]
        processed_memory = np.array([[1.j, 1.j, 1.j],
                                     [1., 1., 1.]], dtype=np.complex_)
        data = models.ExperimentResultData(memory=raw_memory)
        exp_result = models.ExperimentResult(shots=2, success=True, meas_level=0,
                                             meas_return='avg', data=data)
        result = Result(results=[exp_result], **self.base_result_args)
        memory = result.get_memory(0)

        self.assertEqual(memory.shape, (2, 3))
        self.assertEqual(memory.dtype, np.complex_)
        np.testing.assert_almost_equal(memory, processed_memory)

    def test_meas_level_0_single(self):
        """Test measurement level 0 single result."""
        # 3 qubits
        raw_memory = [[[[0., 1.], [0., 1.], [0., 1.]],
                       [[1., 0.], [1., 0.], [1., 0.]]],
                      [[[0., 1.], [0., 1.], [0., 1.]],
                       [[1., 0.], [1., 0.], [1., 0.]]]]
        processed_memory = np.array([[[1.j, 1.j, 1.j],
                                      [1., 1., 1.]],
                                     [[1.j, 1.j, 1.j],
                                      [1., 1., 1.]]], dtype=np.complex_)
        data = models.ExperimentResultData(memory=raw_memory)
        exp_result = models.ExperimentResult(shots=2, success=True, meas_level=0,
                                             meas_return='single', data=data)
        result = Result(results=[exp_result], **self.base_result_args)
        memory = result.get_memory(0)

        self.assertEqual(memory.shape, (2, 2, 3))
        self.assertEqual(memory.dtype, np.complex_)
        np.testing.assert_almost_equal(memory, processed_memory)
