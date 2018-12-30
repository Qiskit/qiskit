# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Custom TestCase for Jobs."""


import time

from qiskit.providers import JobStatus
from qiskit.test import QiskitTestCase


class JobTestCase(QiskitTestCase):
    """Include common functionality when testing jobs."""

    def wait_for_initialization(self, job, timeout=1):
        """Waits until the job progress from `INITIALIZING` to a different
        status.
        """
        waited = 0
        wait = 0.1
        while job.status() is JobStatus.INITIALIZING:
            time.sleep(wait)
            waited += wait
            if waited > timeout:
                self.fail(
                    msg="The JOB is still initializing after timeout ({}s)"
                    .format(timeout)
                )
