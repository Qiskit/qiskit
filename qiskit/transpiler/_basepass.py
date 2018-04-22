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

"""This module implements the base pass.
"""
from abc import ABC, abstractmethod
from qiskit._dagcircuit import DAGCircuit


class BasePass(ABC):
    """Base class for transpiler passes."""
    @abstractmethod
    def __init__(self):
        """Base class for passes.

        This method should initialize the module and its configuration, and
        raise an exception if a component of the module is
        not available.

        Args:
            requires (list[BasePass]): what passes must run before this
            preserves (list[BasePass]): what passes are preserved by this

        Raises:
            FileNotFoundError: if referenced passes are not registered.
        """
        # list of other passes required before this pass can run. this instructs the
        # passmanger to ensure those other passes have run (and not been invalidated)
        # default: the pass is standalone and requires no other pass before it
        self._requires = None

        # list of other passes preserved by this pass. this hints at the passmanager
        # that it is safe to assume those other passes do not need to be rerun
        # default: the pass preseves no other pass; invalidates them all (conservative)
        self._preserves = None

    @abstractmethod
    def run(self, dag):
        """Run a pass on the DAGCircuit.
        """
        return dag

    @property
    def requires(self):
        """Return `requires` list"""
        return self._requires

    @property
    def preserves(self):
        """Return `preserves` list"""
        return self._preserves

