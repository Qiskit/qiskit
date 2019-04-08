# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Tests for the wrapper functionality."""

import os
import unittest

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from qiskit.tools.visualization import HAS_MATPLOTLIB
from qiskit.test import (Path, QiskitTestCase, requires_qe_access, slow_test)


# Timeout (in seconds) for a single notebook.
TIMEOUT = 1000
# Jupyter kernel to execute the notebook in.
JUPYTER_KERNEL = 'python3'


class TestJupyter(QiskitTestCase):
    """Notebooks test case."""
    def setUp(self):
        self.execution_path = os.path.join(Path.SDK.value, '..')

    def _execute_notebook(self, filename, qe_token=None, qe_url=None):
        # Create the preprocessor.
        execute_preprocessor = ExecutePreprocessor(timeout=TIMEOUT,
                                                   kernel_name=JUPYTER_KERNEL)

        # Read the notebook.
        with open(filename) as file_:
            notebook = nbformat.read(file_, as_version=4)

        if qe_token and qe_url:
            top_str = "from qiskit import IBMQ\n"
            top_str += "IBMQ.enable_account('{token}', '{url}')".format(token=qe_token,
                                                                        url=qe_url)
            top = nbformat.notebooknode.NotebookNode({'cell_type': 'code',
                                                      'execution_count': 0,
                                                      'metadata': {},
                                                      'outputs': [],
                                                      'source': top_str})
            notebook.cells = [top] + notebook.cells

        # Run the notebook into the folder containing the `qiskit/` module.
        execute_preprocessor.preprocess(
            notebook, {'metadata': {'path': self.execution_path}})

    def test_jupyter_jobs_pbars(self):
        """Test Jupyter progress bars and job status functionality"""
        self._execute_notebook(self._get_resource_path(
            'notebooks/test_pbar_status.ipynb'))

    @unittest.skipIf(not HAS_MATPLOTLIB, 'matplotlib not available.')
    @requires_qe_access
    @slow_test
    def test_backend_tools(self, qe_token, qe_url):
        """Test Jupyter backend tools."""
        self._execute_notebook(self._get_resource_path(
            'notebooks/test_backend_tools.ipynb'),
                               qe_token=qe_token,
                               qe_url=qe_url)


if __name__ == '__main__':
    unittest.main(verbosity=2)
