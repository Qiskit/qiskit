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

"""This module implements the base pass.
"""
from abc import ABC, abstractmethod


class BasePass(ABC):
    """Base class for passes."""
    @abstractmethod
    def __init__(self, requires=None, preserves=None):
        """Base class for passes.

        This method should initialize the module and its configuration, and
        raise an exception if a component of the module is
        not available.

        Args:
            requires (list): what passes must run before this
            preserves (list): what passes are preserved by this

        Raises:
            FileNotFoundError if referenced passes are not registered.
        """
        self._requires = requires
        self._preserves = preserves

    @abstractmethod
    def run(self, DAGCircuit):
        """Run a pass on the DAGCircuit."""
        pass

    @property
    def requires(self):
        """Return pass requires list"""
        return self._requires

    @property
    def preserves(self):
        """Return pass preserves list"""
        return self._preserves

