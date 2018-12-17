# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Built-in Terra basic Python simulators."""

from .simulatorsprovider import SimulatorsProvider
from .simulatorsjob import SimulatorsJob
from .qasm_simulator import QasmSimulatorPy
from .statevector_simulator import StatevectorSimulatorPy
from .unitary_simulator import UnitarySimulatorPy
from ._simulatorerror import SimulatorError

# Global instance to be used as the entry point for convenience.
BasicAer = SimulatorsProvider()  # pylint: disable=invalid-name
