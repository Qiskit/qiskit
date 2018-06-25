# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""This module implements the base pass.
"""
from abc import ABC, abstractmethod


class BasePass(ABC):
    """Base class for transpiler passes."""

    @abstractmethod
    def run(self, dag):
        """Run a pass on the DAGCircuit.
        """
        return dag
