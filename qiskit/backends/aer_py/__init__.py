# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Aer Backends."""

from .aerpyprovider import AerPyProvider
from .aerpyjob import AerPyJob
from .qasmsimulator import QasmSimulatorPy
from .statevectorsimulator import StatevectorSimulatorPy
from .unitarysimulator import UnitarySimulatorPy

# Global instance to be used as the entry point for convenience.
AerPy = AerPyProvider()  # pylint: disable=invalid-name
