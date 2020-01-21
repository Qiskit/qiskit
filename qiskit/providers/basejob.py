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
from typing import Callable, Optional
import time

from .jobstatus import JOB_FINAL_STATES
from .exceptions import JobTimeoutError
from .basebackend import BaseBackend


class BaseJob(ABC):
    """Class to handle asynchronous jobs"""

    def __init__(self, backend: BaseBackend, job_id: str) -> None:
        """Initializes the asynchronous job.

        Args:
            backend: the backend used to run the job.
            job_id: a unique id in the context of the backend used to run
                the job.
        """
        self._job_id = job_id
        self._backend = backend

    def job_id(self) -> str:
        """Return a unique id identifying the job."""
        return self._job_id

    def backend(self) -> BaseBackend:
        """Return the backend where this job was executed."""
        return self._backend

    def wait_for_final_state(
            self,
            timeout: Optional[float] = None,
            wait: float = 5,
            callback: Optional[Callable] = None
    ) -> None:
        """Poll the job status until it progresses to a final state such as ``DONE`` or ``ERROR``.

        Args:
            timeout: seconds to wait for the job. If ``None``, wait indefinitely. Default: None.
            wait: seconds between queries. Default: 5.
            callback: callback function invoked after each query. Default: None.
                The following positional arguments are provided to the callback function:
                    * job_id: job ID
                    * job_status: status of the job from the last query
                    * job: this BaseJob instance
                Note: different subclass might provider different arguments to
                    the callback function.

        Raises:
            JobTimeoutError: if the job does not reach a final state before the
                specified timeout.
        """
        start_time = time.time()
        status = self.status()
        while status not in JOB_FINAL_STATES:
            elapsed_time = time.time() - start_time
            if timeout is not None and elapsed_time >= timeout:
                raise JobTimeoutError(
                    'Timeout while waiting for job {}.'.format(self.job_id()))
            if callback:
                callback(self.job_id(), status, self)
            time.sleep(wait)
            status = self.status()

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
