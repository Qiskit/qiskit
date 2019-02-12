# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Module containing algorithms used in transpiler pass."""

from .flexlayer_heuristics import FlexlayerHeuristics, remove_head_swaps
from .ancestors import Ancestors
from .dependency_graph import DependencyGraph
