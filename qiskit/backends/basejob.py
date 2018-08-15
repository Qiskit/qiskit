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

    @abstractmethod
    def status(self):
        """Get backend status dictionary"""
        pass
