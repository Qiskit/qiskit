# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Local Backends."""

from .localprovider import LocalProvider
from .localjob import LocalJob
from .qasm_simulator_cpp import CliffordSimulatorCpp, QasmSimulatorCpp
from .qasm_simulator_py import QasmSimulatorPy
from .statevector_simulator_cpp import StatevectorSimulatorCpp
from .statevector_simulator_py import StatevectorSimulatorPy
from .unitary_simulator_py import UnitarySimulatorPy

# Global instance to be used as the entry point for convenience.
Aer = LocalProvider()
