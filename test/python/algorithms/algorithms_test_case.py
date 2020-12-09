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

""" Algorithms Test Case """

import platform
import logging
from qiskit.test import QiskitTestCase


class QiskitAlgorithmsTestCase(QiskitTestCase):
    """Algorithms test Case"""
    def setUp(self):
        super().setUp()
        # disable logging due to Unicode logging error
        if platform.system() == 'Windows':
            self.disable_logging()

    def disable_logging(self):
        """ Disable Qiskit logging"""
        logger = logging.getLogger('qiskit')
        for handler in reversed(logger.handlers):
            self.addCleanup(logger.addHandler, handler)
            logger.removeHandler(handler)
