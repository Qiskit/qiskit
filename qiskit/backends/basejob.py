# -*- coding: utf-8 -*-

# Copyright 2017 IBM RESEARCH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

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
