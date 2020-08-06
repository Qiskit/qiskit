# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=wrong-import-order,invalid-name,wrong-import-position


"""Qiskit integrated unitest handeling."""

import sys
import os

__all__ = ['UnitTester']


class UnitTester:

    """Testrunner class for integrated unitest running."""

    def __init__(self, module_name):
        self.module_name = module_name

    def __call__(self, *args, **kwargs):

        import unittest

        module = sys.modules[self.module_name]
        base_module = sys.modules["qiskit"]
        base_module_path = os.path.abspath(base_module.__path__[0])
        module_path = os.path.relpath(module.__path__[0], base_module_path)
        test_path = os.path.normpath(os.path.join(
            base_module_path, "tests", "python", module_path))

        tests = unittest.defaultTestLoader.discover(test_path)
        text_test_runner = unittest.TextTestRunner(*args, **kwargs)
        text_test_result = text_test_runner.run(tests)

        # return textTestResult.wasSuccessful()
        return len(text_test_result.failures) == 0
