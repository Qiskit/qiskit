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

"""This module implements the job class used by Basic Aer Provider."""

import sys

from qiskit.providers.v2 import Job
from qiskit.result.counts import Counts


class BasicAerJob(Job):
    """BasicAerJob class."""

    def __init__(self, job_id, backend, result_data, time_taken):
        super().__init__(job_id, backend, time_taken=time_taken)
        self.result_data = result_data

    def status(self):
        return 'COMPLETE'

    def wait_for_final_state(self):
        return True

