# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""This module implements the abstract base class for backend jobs

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
        """Return the backend for this job."""
        return self._backend

    @abstractmethod
    def submit(self):
        """Submit the job to the backend."""
        pass

    @abstractmethod
    def result(self):
        """Return backend result"""
        pass

    @abstractmethod
    def cancel(self):
        """Attempt to cancel job."""
        pass

    @abstractmethod
    def status(self):
        """Return one of the values of ``JobStatus``"""
        pass
