# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""Test Qiskit's Result class."""

from qiskit.result import Result
from qiskit import QiskitError
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
