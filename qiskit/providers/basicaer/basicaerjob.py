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

import warnings

from qiskit.providers import JobStatus
from qiskit.providers.job import JobV1


class BasicAerJob(JobV1):
    """BasicAerJob class."""

    _async = False

    def __init__(self, backend, job_id, result):
        super().__init__(backend, job_id)
        self._result = result

    def submit(self):
        """Submit the job to the backend for execution.

        Raises:
            JobError: if trying to re-submit the job.
        """
        return

    def result(self, timeout=None):
        # pylint: disable=arguments-differ
        """Get job result .

        Returns:
            qiskit.Result: Result object
        """
        if timeout is not None:
            warnings.warn(
                "The timeout kwarg doesn't have any meaning with "
                "BasicAer because execution is synchronous and the "
                "result already exists when run() returns.",
                UserWarning,
            )

        return self._result

    def status(self):
        """Gets the status of the job by querying the Python's future

        Returns:
            qiskit.providers.JobStatus: The current JobStatus
        """
        return JobStatus.DONE

    def backend(self):
        """Return the instance of the backend used for this job."""
        return self._backend
