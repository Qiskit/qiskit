# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Shared functionality and helpers for the unit tests."""

# pylint: disable=unused-import

# TODO: once all the tests in test/python import from qiskit.test, this file
# can be safely removed.

from qiskit.test.base import QiskitTestCase
from qiskit.test.decorators import (requires_cpp_simulator, requires_qe_access,
                                    slow_test, is_cpp_simulator_available)
from qiskit.test.utils import Path
