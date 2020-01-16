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

"""This module implements the abstract base class for backend jobs.

When creating a new backend module it is also necessary to implement this
job interface.
"""

from abc import ABC, abstractmethod


class BaseJob(ABC):
    """Class to handle asynchronous jobs"""

    def __init__(self, backend, job_id):
        """Initializes the asynchronous job.

        Args:
            backend (BaseBackend): the backend used to run the job.
            job_id (str): a unique id in the context of the backend used to run
                the job.
        """
        self._job_id = job_id
        self._backend = backend

    def job_id(self):
        """Return a unique id identifying the job."""
        return self._job_id

    def backend(self):
        """Return the backend where this job was executed."""
        return self._backend

    def done(self):
        """Return whether the job has successfully run.

        Returns:
             True if job status is done, else false.
         """
        return self.status() == JobStatus.DONE

    def running(self):
        """Return whether the job is actively running.

        Returns:
             True if job status is running, else false.
         """
        return self.status() == JobStatus.RUNNING

    def cancelled(self):
        """Return whether the job has been cancelled.

        Returns:
            True if job status is cancelled, else false.
        """
        return self.status() == JobStatus.CANCELLED

    @abstractmethod
    def submit(self):
        """Submit the job to the backend for execution."""
        pass

    @abstractmethod
    def result(self):
        """Return the results of the job."""
        pass

    @abstractmethod
    def cancel(self):
        """Attempt to cancel the job."""
        pass

    @abstractmethod
    def status(self):
        """Return the status of the job, among the values of ``JobStatus``."""
        pass
