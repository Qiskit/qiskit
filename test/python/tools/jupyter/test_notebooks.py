# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=bad-docstring-quotes

"""Tests for the wrapper functionality."""

import os
import sys
import unittest

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from qiskit.tools.visualization import HAS_MATPLOTLIB
from qiskit.test import (Path, QiskitTestCase, slow_test)


# Timeout (in seconds) for a single notebook.
TIMEOUT = 1000
# Jupyter kernel to execute the notebook in.
JUPYTER_KERNEL = 'python3'


class TestJupyter(QiskitTestCase):
    """Notebooks test case."""
    def setUp(self):
        self.execution_path = os.path.join(Path.SDK.value, '..')

    def _execute_notebook(self, filename):
        # Create the preprocessor.
        execute_preprocessor = ExecutePreprocessor(timeout=TIMEOUT,
                                                   kernel_name=JUPYTER_KERNEL)

        # Read the notebook.
        with open(filename) as file_:
            notebook = nbformat.read(file_, as_version=4)

        top_str = """
        import qiskit
        import sys
        from unittest.mock import create_autospec, MagicMock
        from qiskit.test.mock import FakeProviderFactory
        from qiskit.providers import basicaer
        fake_prov = FakeProviderFactory()
        qiskit.IBMQ = fake_prov
        ibmq_mock = create_autospec(basicaer)
        ibmq_mock.IBMQJobApiError = MagicMock()
        sys.modules['qiskit.providers.ibmq'] = ibmq_mock
        sys.modules['qiskit.providers.ibmq.job'] = ibmq_mock
        sys.modules['qiskit.providers.ibmq.job.exceptions'] = ibmq_mock
        """
        top = nbformat.notebooknode.NotebookNode({'cell_type': 'code',
                                                  'execution_count': 0,
                                                  'metadata': {},
                                                  'outputs': [],
                                                  'source': top_str})
        notebook.cells = [top] + notebook.cells

        # Run the notebook into the folder containing the `qiskit/` module.
        execute_preprocessor.preprocess(
            notebook, {'metadata': {'path': self.execution_path}})

    @unittest.skipIf(
        sys.version_info[0] == 3 and sys.version_info[1] == 8 and
        sys.platform != 'linux', 'Fails with Python 3.8 on osx and windows')
    def test_jupyter_jobs_pbars(self):
        """Test Jupyter progress bars and job status functionality"""
        self._execute_notebook(self._get_resource_path(
            'notebooks/test_pbar_status.ipynb'))

    @unittest.skipIf(not HAS_MATPLOTLIB, 'matplotlib not available.')
    @slow_test
    def test_backend_tools(self):
        """Test Jupyter backend tools."""
        self._execute_notebook(self._get_resource_path(
            'notebooks/test_backend_tools.ipynb'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
