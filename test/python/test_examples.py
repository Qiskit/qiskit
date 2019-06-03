# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# pylint: disable=invalid-name

"""Test examples scripts."""

import os
import subprocess
import sys

import ddt

from qiskit.test import QiskitTestCase, online_test

examples_dir = os.path.abspath(os.path.join(
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                 'examples'),
    'python'))
ibmq_examples_dir = os.path.join(examples_dir, 'ibmq')
examples = []
ibmq_examples = []
if os.path.isdir(examples_dir):
    examples = [x for x in os.listdir(examples_dir) if x.endswith('.py')]

if os.path.isdir(ibmq_examples_dir):
    ibmq_examples = [
        x for x in os.listdir(ibmq_examples_dir) if x.endswith('.py')]


@ddt.ddt
class TestPythonExamples(QiskitTestCase):
    """Test example scripts"""
    @ddt.data(*examples)
    def test_all_examples(self, example):
        """Execute the example python files and pass if it returns 0."""
        example_path = os.path.join(examples_dir, example)
        cmd = [sys.executable, example_path]
        run_example = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)
        stdout, stderr = run_example.communicate()
        error_string = "Running example %s failed with return code %s\n" % (
            example, run_example.returncode)
        error_string += "\tstdout:%s\n\tstderr: %s" % (stdout, stderr)
        self.assertEqual(run_example.returncode, 0, error_string)

    @online_test
    @ddt.data(*ibmq_examples)
    def test_all_ibmq_examples(self, example, qe_token, qe_url):
        """Execute the ibmq example python files and pass if it returns 0."""
        from qiskit import IBMQ  # pylint: disable: import-error
        IBMQ.save_account(qe_token, qe_url)
        self.addCleanup(IBMQ.delete_accounts, token=qe_token, url=qe_url)
        example_path = os.path.join(ibmq_examples_dir, example)
        cmd = [sys.executable, example_path]
        run_example = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)
        stdout, stderr = run_example.communicate()
        error_string = "Running example %s failed with return code %s\n" % (
            example, run_example.returncode)
        error_string += "\tstdout:%s\n\tstderr: %s" % (stdout, stderr)
        self.assertEqual(run_example.returncode, 0, error_string)
