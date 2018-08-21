# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""This module implements the job class used for LocalBackend objects."""

from concurrent import futures
import json
import logging
import os
import sys
import functools
import jsonschema

from qiskit.backends import BaseJob, JobStatus, JobError
from qiskit import __path__ as qiskit_path

logger = logging.getLogger(__name__)


def requires_submit(func):
    """
    Decorator to ensure that a submit has been performed before
    calling the method.

    Args:
        func (callable): test function to be decorated.

    Returns:
        callable: the decorated function.
    """
    @functools.wraps(func)
    def _wrapper(self, *args, **kwargs):
        if self._future is None:
            raise JobError("Job not submitted yet!. You have to .submit() first!")
        return func(self, *args, **kwargs)
    return _wrapper


class LocalJob(BaseJob):
    """Local QISKit SDK Job class.

    Attributes:
        _executor (futures.Executor): executor to handle asynchronous jobs
    """

    if sys.platform in ['darwin', 'win32']:
        _executor = futures.ThreadPoolExecutor()
    else:
        _executor = futures.ProcessPoolExecutor()

    def __init__(self, fn, qobj):
        super().__init__()
        self._fn = fn
        self._qobj = qobj
        self._backend_name = qobj.header.backend_name
        self._future = None

        # verify the QObj is valid before making requests for computing resources
        sdk = qiskit_path[0]
        # Schemas path:     qiskit/backends/schemas
        schemas_path = os.path.join(sdk, 'schemas')
        schema_file_path = os.path.join(schemas_path, 'qobj_schema.json')

        with open(schema_file_path, 'r') as schema_file:
            schema = json.load(schema_file)

        try:
            jsonschema.validate(self._qobj.as_dict(), schema)
        except jsonschema.ValidationError as validation_error:
            raise ValueError(validation_error)

    def submit(self):
        """Submit the job to the backend for running """
        if self._future is not None:
            raise JobError("We have already submitted the job!")

        self._future = self._executor.submit(self._fn, self._qobj)

    @requires_submit
    def result(self, timeout=None):
        # pylint: disable=arguments-differ
        """Get job result. The behavior is the same as the underlying
        concurrent Future objects,

        https://docs.python.org/3/library/concurrent.futures.html#future-objects

        Args:
            timeout (float): number of seconds to wait for results.

        Returns:
            Result: Result object

        Raises:
            concurrent.futures.TimeoutError: if timeout occurred.
            concurrent.futures.CancelledError: if job cancelled before completed.
        """
        return self._future.result(timeout=timeout)

    @requires_submit
    def cancel(self):
        return self._future.cancel()

    @requires_submit
    def status(self):
        """Gets the status of the job by querying the Python's future

        Returns:
            JobStatus: The current JobStatus

        Raises:
            JobError: If the future is in unexpected state
            concurrent.futures.TimeoutError: if timeout occurred.
        """
        # The order is important here
        if self._future.running():
            _status = JobStatus.RUNNING
        elif self._future.cancelled():
            _status = JobStatus.CANCELLED
        elif self._future.done():
            _status = JobStatus.DONE if self._future.exception() is None else JobStatus.ERROR
        else:
            raise JobError('Unexpected behavior of {0}'.format(
                self.__class__.__name__))
        return _status

    @property
    def backend_name(self):
        """
        Return backend name used for this job
        """
        return self._backend_name
