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

"""This module implements the legacy abstract base class for backend jobs.

When creating a new backend module it is also necessary to implement this
job interface.
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional
import time
import warnings

from .jobstatus import JobStatus, JOB_FINAL_STATES
from .exceptions import JobTimeoutError
from .basebackend import BaseBackend


class BaseJob(ABC):
    """DEPRECATED Legacy Class to handle asynchronous jobs"""

    def __init__(self, backend: BaseBackend, job_id: str) -> None:
        """Initializes the asynchronous job.

        Args:
            backend: the backend used to run the job.
            job_id: a unique id in the context of the backend used to run
                the job.
        """
        warnings.warn(
            "The BaseJob abstract interface is deprecated as of "
            "the 0.18.0 release and will be removed in a future "
            "release. Instead you should build your backends using "
            "the JobV1 abstract class (which is the current "
            "latest version of the backend interface)",
            DeprecationWarning,
            stacklevel=2,
        )
        self._job_id = job_id
        self._backend = backend

    def job_id(self) -> str:
        """Return a unique id identifying the job."""
        return self._job_id

    def backend(self) -> BaseBackend:
        """Return the backend where this job was executed."""
        return self._backend

    def done(self) -> bool:
        """Return whether the job has successfully run."""
        return self.status() == JobStatus.DONE

    def running(self) -> bool:
        """Return whether the job is actively running."""
        return self.status() == JobStatus.RUNNING

    def cancelled(self) -> bool:
        """Return whether the job has been cancelled."""
        return self.status() == JobStatus.CANCELLED

    def in_final_state(self) -> bool:
        """Return whether the job is in a final job state."""
        return self.status() in JOB_FINAL_STATES

    def wait_for_final_state(
        self, timeout: Optional[float] = None, wait: float = 5, callback: Optional[Callable] = None
    ) -> None:
        """Poll the job status until it progresses to a final state such as ``DONE`` or ``ERROR``.

        Args:
            timeout: Seconds to wait for the job. If ``None``, wait indefinitely.
            wait: Seconds between queries.
            callback: Callback function invoked after each query.
                The following positional arguments are provided to the callback function:

                * job_id: Job ID
                * job_status: Status of the job from the last query
                * job: This BaseJob instance

                Note: different subclass might provide different arguments to
                the callback function.

        Raises:
            JobTimeoutError: If the job does not reach a final state before the
                specified timeout.
        """
        start_time = time.time()
        status = self.status()
        while status not in JOB_FINAL_STATES:
            elapsed_time = time.time() - start_time
            if timeout is not None and elapsed_time >= timeout:
                raise JobTimeoutError(f"Timeout while waiting for job {self.job_id()}.")
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
