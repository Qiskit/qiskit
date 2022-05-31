# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
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

from qiskit.providers.fake_provider.fake_provider import FakeProviderForBackendV2, FakeProvider
from qiskit.providers.fake_provider.fake_provider import FakeProviderFactory
from qiskit.providers.fake_provider.fake_backend import FakeBackend
from qiskit.providers.fake_provider.fake_pulse_backend import FakePulseBackend
from qiskit.providers.fake_provider.fake_qasm_backend import FakeQasmBackend
from qiskit.providers.fake_provider.utils.configurable_backend import ConfigurableFakeBackend
from qiskit.providers.fake_provider.fake_mumbai_v2 import FakeMumbaiFractionalCX
from qiskit.providers.fake_provider.fake_job import FakeJob
from qiskit.providers.fake_provider.fake_qobj import FakeQobj

from qiskit.providers.fake_provider.backends import *

from qiskit.providers.fake_provider.fake_qasm_simulator import FakeQasmSimulator
from qiskit.providers.fake_provider.fake_openpulse_2q import FakeOpenPulse2Q
from qiskit.providers.fake_provider.fake_openpulse_3q import FakeOpenPulse3Q
from qiskit.providers.fake_provider.fake_1q import Fake1Q
