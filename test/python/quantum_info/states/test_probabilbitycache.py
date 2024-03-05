# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Tests for ProbabilityCache quantum state class."""

import unittest
import logging
from ddt import ddt

# from qiskit.quantum_info.states.probabilitycache import ProbabilityCache
# from test import combine  # pylint: disable=wrong-import-order
from test import QiskitTestCase  # pylint: disable=wrong-import-order

logger = logging.getLogger(__name__)


@ddt
class TestProbabilityCache(QiskitTestCase):
    """Tests for StabilizerState class."""


if __name__ == "__main__":
    unittest.main()
