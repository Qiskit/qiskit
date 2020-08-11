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
from qiskit.result import marginal_counts
from qiskit.result import Result
from qiskit.qobj import QobjExperimentHeader
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
        data = models.ExperimentResultData(counts=dict(**raw_counts))
        exp_result = models.ExperimentResult(shots=14, success=True, meas_level=2, data=data)
        result = Result(results=[exp_result], **self.base_result_args)

        self.assertEqual(result.get_counts(0), no_header_processed_counts)

    def test_counts_header(self):
        """Test that counts are extracted properly with header."""
        raw_counts = {'0x0': 4, '0x2': 10}
        processed_counts = {'0 0 00': 4, '0 0 10': 10}
        data = models.ExperimentResultData(counts=dict(**raw_counts))
        exp_result_header = QobjExperimentHeader(
            creg_sizes=[['c0', 2], ['c0', 1], ['c1', 1]], memory_slots=4)
        exp_result = models.ExperimentResult(shots=14, success=True, meas_level=2,
                                             data=data, header=exp_result_header)
        result = Result(results=[exp_result], **self.base_result_args)

        self.assertEqual(result.get_counts(0), processed_counts)

    def test_multiple_circuits_counts(self):
        """"
        Test that counts are returned either as a list or a single item.

        Counts are returned as a list when multiple experiments are executed
        and get_counts() is called with no arguments. In all the other cases
        get_counts() returns a single item containing the counts for a
        single experiment.
        """
        raw_counts_1 = {'0x0': 5, '0x3': 12, '0x5': 9, '0xD': 6, '0xE': 2}
        processed_counts_1 = {'0000': 5, '0011': 12, '0101': 9, '1101': 6, '1110': 2}
        data_1 = models.ExperimentResultData(counts=dict(**raw_counts_1))
        exp_result_header_1 = QobjExperimentHeader(creg_sizes=[['c0', 4]], memory_slots=4)
        exp_result_1 = models.ExperimentResult(shots=14, success=True, meas_level=2, data=data_1,
                                               header=exp_result_header_1)

        raw_counts_2 = {'0x1': 0, '0x4': 3, '0x6': 6, '0xA': 1, '0xB': 2}
        processed_counts_2 = {'0001': 0, '0100': 3, '0110': 6, '1010': 1, '1011': 2}
        data_2 = models.ExperimentResultData(counts=dict(**raw_counts_2))
        exp_result_header_2 = QobjExperimentHeader(creg_sizes=[['c0', 4]], memory_slots=4)
        exp_result_2 = models.ExperimentResult(shots=14, success=True, meas_level=2, data=data_2,
                                               header=exp_result_header_2)

        raw_counts_3 = {'0xC': 27, '0xF': 20}
        processed_counts_3 = {'1100': 27, '1111': 20}
        data_3 = models.ExperimentResultData(counts=dict(**raw_counts_3))
        exp_result_header_3 = QobjExperimentHeader(creg_sizes=[['c0', 4]], memory_slots=4)
        exp_result_3 = models.ExperimentResult(shots=14, success=True, meas_level=2, data=data_3,
                                               header=exp_result_header_3)

        mult_result = Result(results=[exp_result_1, exp_result_2, exp_result_3],
                             **self.base_result_args)
        sing_result = Result(results=[exp_result_1], **self.base_result_args)

        self.assertEqual(mult_result.get_counts(), [processed_counts_1, processed_counts_2,
                                                    processed_counts_3])
        self.assertEqual(sing_result.get_counts(), processed_counts_1)

    def test_marginal_counts(self):
        """Test that counts are marginalized correctly."""
        raw_counts = {'0x0': 4, '0x1': 7, '0x2': 10, '0x6': 5, '0x9': 11, '0xD': 9, '0xE': 8}
        data = models.ExperimentResultData(counts=dict(**raw_counts))
        exp_result_header = QobjExperimentHeader(creg_sizes=[['c0', 4]],
                                                 memory_slots=4)
        exp_result = models.ExperimentResult(shots=54, success=True, data=data,
                                             header=exp_result_header)
        result = Result(results=[exp_result], **self.base_result_args)
        expected_marginal_counts = {'00': 4, '01': 27, '10': 23}

        self.assertEqual(marginal_counts(result.get_counts(), [0, 1]), expected_marginal_counts)
        self.assertEqual(marginal_counts(result.get_counts(), [1, 0]), expected_marginal_counts)

    def test_marginal_counts_result(self):
        """Test that a Result object containing counts marginalizes correctly."""
        raw_counts_1 = {'0x0': 4, '0x1': 7, '0x2': 10, '0x6': 5, '0x9': 11, '0xD': 9, '0xE': 8}
        data_1 = models.ExperimentResultData(counts=dict(**raw_counts_1))
        exp_result_header_1 = QobjExperimentHeader(creg_sizes=[['c0', 4]], memory_slots=4)
        exp_result_1 = models.ExperimentResult(shots=54, success=True, data=data_1,
                                               header=exp_result_header_1)

        raw_counts_2 = {'0x2': 5, '0x3': 8}
        data_2 = models.ExperimentResultData(counts=dict(**raw_counts_2))
        exp_result_header_2 = QobjExperimentHeader(creg_sizes=[['c0', 2]], memory_slots=2)
        exp_result_2 = models.ExperimentResult(shots=13, success=True, data=data_2,
                                               header=exp_result_header_2)

        result = Result(results=[exp_result_1, exp_result_2], **self.base_result_args)

        expected_marginal_counts_1 = {'00': 4, '01': 27, '10': 23}
        expected_marginal_counts_2 = {'0': 5, '1': 8}

        self.assertEqual(marginal_counts(result, [0, 1]).get_counts(0),
                         expected_marginal_counts_1)
        self.assertEqual(marginal_counts(result, [0]).get_counts(1),
                         expected_marginal_counts_2)

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
        exp_result_header = QobjExperimentHeader(
            creg_sizes=[['c0', 2], ['c0', 1], ['c1', 1]], memory_slots=4)
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
