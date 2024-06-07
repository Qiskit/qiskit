# This code is part of Qiskit.
#
# (C) Copyright IBM 2022, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Job implementation for the reference implementations of Primitives.
"""

import uuid
from concurrent.futures import ThreadPoolExecutor

from qiskit.providers import JobError, JobStatus
from qiskit.providers.jobstatus import JOB_FINAL_STATES

from .base.base_primitive_job import BasePrimitiveJob, ResultT


class PrimitiveJob(BasePrimitiveJob[ResultT, JobStatus]):
    """
    Primitive job class for the reference implementations of Primitives.
    """

    def __init__(self, function, *args, **kwargs):
        """
        Args:
            function: A callable function to execute the job.
        """
        super().__init__(str(uuid.uuid4()))
        self._future = None
        self._function = function
        self._args = args
        self._kwargs = kwargs

    def _submit(self):
        if self._future is not None:
            raise JobError("Primitive job has been submitted already.")

        executor = ThreadPoolExecutor(max_workers=1)  # pylint: disable=consider-using-with
        self._future = executor.submit(self._function, *self._args, **self._kwargs)
        executor.shutdown(wait=False)

    def result(self) -> ResultT:
        self._check_submitted()
        return self._future.result()

    def status(self) -> JobStatus:
        self._check_submitted()
        if self._future.running():
            return JobStatus.RUNNING
        elif self._future.cancelled():
            return JobStatus.CANCELLED
        elif self._future.done() and self._future.exception() is None:
            return JobStatus.DONE
        return JobStatus.ERROR

    def _check_submitted(self):
        if self._future is None:
            raise JobError("Primitive Job has not been submitted yet.")

    def cancel(self):
        self._check_submitted()
        return self._future.cancel()

    def done(self) -> bool:
        return self.status() == JobStatus.DONE

    def running(self) -> bool:
        return self.status() == JobStatus.RUNNING

    def cancelled(self) -> bool:
        return self.status() == JobStatus.CANCELLED

    def in_final_state(self) -> bool:
        return self.status() in JOB_FINAL_STATES
