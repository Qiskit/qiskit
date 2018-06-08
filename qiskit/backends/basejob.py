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
import enum


class BaseJob(ABC):
    """Class to handle asynchronous jobs"""

    @abstractmethod
    def __init__(self):
        """Initializes the asynchronous job"""
        pass

    @abstractmethod
    def result(self):
        """Return backend result"""
        pass

    @abstractmethod
    def cancel(self):
        """Attempt to cancel job."""
        pass

    # Property attributes
    #####################
    @property
    @abstractmethod
    def status(self):
        """Get backend status dictionary"""
        pass

    @property
    @abstractmethod
    def running(self):
        """True if job is currently running."""
        pass

    @property
    @abstractmethod
    def done(self):
        """True if call was successfully finished."""
        pass

    @property
    @abstractmethod
    def cancelled(self):
        """True if call was successfully cancelled"""
        pass


class JobStatus(enum.Enum):
    """Class for job status enumerated type."""
    INITIALIZING = 'job is being initialized'
    QUEUED = 'job is queued'
    RUNNING = 'job is actively running'
    CANCELLED = 'job has been cancelled'
    DONE = 'job has successfully run'
    ERROR = 'job incurred error'
