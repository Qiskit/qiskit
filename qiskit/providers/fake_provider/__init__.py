# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Utilities for mocking the IBMQ provider, including job responses and backends.

The module includes dummy provider, backends, and jobs.
The purpose of these classes is to fake backends for testing purposes:
testing local timeouts, arbitrary responses or behavior, etc.

The mock devices are mainly for testing the compiler.
"""

# Fake job and qobj classes
from .fake_job import FakeJob
from .fake_qobj import FakeQobj

# Base classes for fake backends
from .fake_backend import FakeBackend
from .fake_pulse_backend import FakePulseBackend
from .fake_qasm_backend import FakeQasmBackend

# Fake providers
from .fake_provider import FakeProviderFactory, FakeProviderForBackendV2, FakeProvider

# Standard fake backends with IBM Quantum systems snapshots
from .backends import *

# Special fake backends for special testing perpurposes
from .fake_qasm_simulator import FakeQasmSimulator
from .fake_openpulse_2q import FakeOpenPulse2Q
from .fake_openpulse_3q import FakeOpenPulse3Q
from .fake_1q import Fake1Q
from .fake_mumbai_v2 import FakeMumbaiFractionalCX

# Configurable fake backend
from .utils.configurable_backend import ConfigurableFakeBackend
