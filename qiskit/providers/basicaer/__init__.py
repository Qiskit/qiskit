# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""BasicAer Provider: Contains Python simulators."""

from .basicaerprovider import BasicAerProvider
from .basicaerjob import BasicAerJob
from .qasm_simulator import QasmSimulatorPy
from .statevector_simulator import StatevectorSimulatorPy
from .unitary_simulator import UnitarySimulatorPy
from .exceptions import BasicAerError

# Global instance to be used as the entry point for convenience.
BasicAer = BasicAerProvider()  # pylint: disable=invalid-name
