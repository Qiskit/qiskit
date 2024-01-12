# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
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
from typing import TypeVar, Union

from qiskit.providers import JobError, JobStatus
from qiskit.utils.deprecation import deprecate_func

from .base.base_primitive_job import BasePrimitiveJob
from .base.base_result import BasePrimitiveResult
from .containers import PrimitiveResult

Result = TypeVar("Result", bound=Union[BasePrimitiveResult, PrimitiveResult])


class PrimitiveJob(BasePrimitiveJob[Result, JobStatus]):
    """
    PrimitiveJob class for the reference implementations of Primitives.
    """

    def __init__(self, function, *args, **kwargs):
        """
        Args:
            function: a callable function to execute the job.
        """
        self._job_id = str(uuid.uuid4())
        self._future = None
        self._function = function
        self._args = args
        self._kwargs = kwargs

    @deprecate_func(since="0.46.0")
    def submit(self):
        """Submit a job.

        .. deprecated:: 0.46.0
            ``submit`` method is deprecated as of Qiskit 0.46 and will be removed
            no earlier than 3 months after the release date.

        """
        self._submit()

    def _submit(self):
        if self._future is not None:
            raise JobError("Primitive job has already been submitted.")

        executor = ThreadPoolExecutor(max_workers=1)  # pylint: disable=consider-using-with
        self._future = executor.submit(self._function, *self._args, **self._kwargs)
        executor.shutdown(wait=False)

    def result(self) -> Result:
        """Return the results of the job."""
        self._check_submitted()
        return self._future.result()

    def cancel(self):
        """Cancel the job."""
        self._check_submitted()
        return self._future.cancel()

    def status(self) -> JobStatus:
        """Return the status of the job."""
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
            raise JobError("Job not submitted yet!. You have to .submit() first!")
