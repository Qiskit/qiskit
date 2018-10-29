# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=redefined-builtin

"""Tests for the wrapper functionality."""

import os
import unittest
import subprocess
import tempfile
from .common import QiskitTestCase


def _exec_notebook(path):
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as fout:
        args = ["jupyter", "nbconvert", "--to", "notebook", "--execute",
                "--ExecutePreprocessor.timeout=1000",
                "--output", fout.name, path]
    subprocess.check_call(args)


class TestJupyter(QiskitTestCase):
    """Notebooks test case."""
    def setUp(self):
        self.path = os.path.dirname(os.path.realpath(__file__))

    def test_jupyter(self):
        "Test Jupyter functionality"
        success = True
        try:
            _exec_notebook(self.path+'/notebooks/test_jupyter.ipynb')
        except Exception as excep:  # pylint: disable=w0703
            success = False
            print(excep)
        self.assertTrue(success)


if __name__ == '__main__':
    unittest.main(verbosity=2)
