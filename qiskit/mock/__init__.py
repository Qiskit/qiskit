# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
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

from ..test.mock.fake_provider import FakeProvider, FakeLegacyProvider
from ..test.mock.fake_provider import FakeProviderFactory
from ..test.mock.fake_backend import FakeBackend, FakeLegacyBackend, HAS_AER
from ..test.mock.fake_pulse_backend import FakePulseBackend, FakePulseLegacyBackend
from ..test.mock.fake_qasm_backend import FakeQasmBackend, FakeQasmLegacyBackend
from ..test.mock.utils.configurable_backend import ConfigurableFakeBackend
from ..test.mock.fake_backend_v2 import FakeBackendV2, FakeBackend5QV2
from ..test.mock.fake_mumbai_v2 import FakeMumbaiV2
from ..test.mock.fake_job import FakeJob, FakeLegacyJob
from ..test.mock.fake_qobj import FakeQobj

from ..test.mock.backends import *

from ..test.mock.fake_qasm_simulator import FakeQasmSimulator
from ..test.mock.fake_openpulse_2q import FakeOpenPulse2Q
from ..test.mock.fake_openpulse_3q import FakeOpenPulse3Q
from ..test.mock.fake_1q import Fake1Q
