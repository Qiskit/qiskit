# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""This module defines an enumerated type for the state of backend jobs"""

import enum


class JobStatus(enum.Enum):
    """Class for job status enumerated type."""
    INITIALIZING = 'job is being initialized'
    QUEUED = 'job is queued'
    VALIDATING = 'job is being validated'
    RUNNING = 'job is actively running'
    CANCELLED = 'job has been cancelled'
    DONE = 'job has successfully run'
    ERROR = 'job incurred error'


JOB_FINAL_STATES = (
    JobStatus.DONE,
    JobStatus.CANCELLED,
    JobStatus.ERROR
)
