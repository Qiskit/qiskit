# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=redefined-builtin

"""Tests for the wrapper functionality."""

import os
import unittest

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

from ...common import Path, QiskitTestCase, requires_cpp_simulator


# Timeout (in seconds) for a single notebook.
TIMEOUT = 1000
# Jupyter kernel to execute the notebook in.
JUPYTER_KERNEL = 'python3'


class TestJupyter(QiskitTestCase):
    """Notebooks test case."""
    def setUp(self):
        self.filename = self._get_resource_path(
            'notebooks/test_pbar_status.ipynb')
        self.execution_path = os.path.join(Path.SDK.value, '..')

    def _execute_notebook(self, filename):
        # Create the preprocessor.
        execute_preprocessor = ExecutePreprocessor(timeout=TIMEOUT,
                                                   kernel_name=JUPYTER_KERNEL)

        # Read the notebook.
        with open(filename) as file_:
            notebook = nbformat.read(file_, as_version=4)

        # Run the notebook into the folder containing the `qiskit/` module.
        execute_preprocessor.preprocess(
            notebook, {'metadata': {'path': self.execution_path}})

    @requires_cpp_simulator
    def test_jupyter(self):
        "Test Jupyter functionality"
        self._execute_notebook(self.filename)


if __name__ == '__main__':
    unittest.main(verbosity=2)
