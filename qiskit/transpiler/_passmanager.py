# -*- coding: utf-8 -*-

# Copyright 2018 IBM RESEARCH. All Rights Reserved.
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

"""This module implements a passmanager 
"""
from ._basepass import BasePass

class PassManager():
    """PassManager class for the transpiler."""
    def __init__(self, configuration=None):
        """Base class for backends.

        This method should initialize the module and its configuration, and
        raise an exception if a component of the module is
        not available.

        Args:
            configuration (dict): configuration dictionary
        """
        self._configuration = configuration
        self._resources = {}

    def add(self, BasePass):
        """Schedule a pass in the passmanager."""
        pass

    def run(self, BasePass, DAGCircuit):
        """Run a Pass on the DAGCircuit."""
        pass

    @property
    def configuration(self):
        """Return backend configuration"""
        return self._configuration
