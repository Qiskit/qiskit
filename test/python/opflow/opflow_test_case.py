# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Opflow Test Case"""

import warnings
from qiskit.test import QiskitTestCase


class QiskitOpflowTestCase(QiskitTestCase):
    """Opflow test Case"""

    def setUp(self):
        super().setUp()
        # ignore opflow msgs
        warnings.filterwarnings("ignore", category=DeprecationWarning, message=r".*opflow.*")

    def tearDown(self):
        super().tearDown()
        # restore opflow msgs
        warnings.filterwarnings("error", category=DeprecationWarning, message=r".*opflow.*")
