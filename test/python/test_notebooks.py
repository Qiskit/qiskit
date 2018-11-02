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

    @unittest.skipIf(os.getenv('APPVEYOR', None), 'Cannot make temp file in Appveyor.')
    def test_jupyter(self):
        "Test Jupyter functionality"
        _exec_notebook(self.path+'/notebooks/test_jupyter.ipynb')


if __name__ == '__main__':
    unittest.main(verbosity=2)
