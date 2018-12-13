# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Built-in Terra Simulators."""

from .simulatorsprovider import SimulatorsProvider
from .simulatorsjob import SimulatorsJob
from .qasm_simulator_py import QasmSimulatorPy
from .statevector_simulator_py import StatevectorSimulatorPy
from .unitary_simulator_py import UnitarySimulatorPy
from ._simulatorerror import SimulatorError

# Global instance to be used as the entry point for convenience.
Simulators = SimulatorsProvider()  # pylint: disable=invalid-name
