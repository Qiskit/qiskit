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

"""This module implements the abstract base class for backend modules.

To create add-on backend modules subclass the Backend class in this module.
Doing so requires that the required backend interface is implemented.
"""

from abc import ABC, abstractmethod
from qiskit._qiskiterror import QISKitError


class BaseBackend(ABC):
    """Base class for backends."""

    @abstractmethod
    def __init__(self, configuration):
        """Base class for backends.

        This method should initialize the module and its configuration, and
        raise an exception if a component of the module is
        not available.

        Args:
            configuration (dict): configuration dictionary

        Raises:
            FileNotFoundError if backend executable is not available.
            QISKitError: if there is no name in the configuration
        """
        if 'name' not in configuration:
            raise QISKitError('backend does not have a name.')
        self._configuration = configuration

    @abstractmethod
    def run(self, q_job):
        """Run a QuantumJob on the the backend."""
        pass

    @property
    def configuration(self):
        """Return backend configuration"""
        return self._configuration

    @property
    def calibration(self):
        """Return backend calibration"""
        return {}

    @property
    def parameters(self):
        """Return backend parameters"""
        return {}

    @property
    def status(self):
        """Return backend status"""
        return {'name': self.name, 'available': True}

    @property
    def name(self):
        """Return backend name"""
        return self._configuration['name']

    def __str__(self):
        return self.name
