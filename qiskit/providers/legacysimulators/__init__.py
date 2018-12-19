# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Legacy C++ simulators from Qiskit versions prior to 0.7"""

from .legacyprovider import LegacyProvider
from .qasm_simulator import QasmSimulator, CliffordSimulator
from .statevector_simulator import StatevectorSimulator

LegacySimulators = LegacyProvider()  # pylint: disable=invalid-name
