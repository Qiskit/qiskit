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
======================================================
Fake Provider (:mod:`qiskit.providers.fake_provider`)
======================================================

.. currentmodule:: qiskit.providers.fake_provider

Overview
========

The FakeProvider class contains fake providers and fake backends, primarily used for unit testing of
the transpiler.

Fake Providers
==============

.. autosummary::
    :toctree: ../stubs/

    FakeProviderFactory
    FakeProviderForBackendV2
    FakeProvider

Fake Backends
=============

V2 Backends
-----------

.. autosummary::
    :toctree: ../stubs/

    FakeAlmadenV2
    FakeArmonkV2

V1 Backends
-----------

.. autosummary::
    :toctree: ../stubs/

    FakeAlmaden
    FakeArmonk

Special Fake Backends
=====================

.. autosummary::
    :toctree: ../stubs/
    FakeQasmSimulator

"""

# Fake job and qobj classes
from .fake_job import FakeJob
from .fake_qobj import FakeQobj

# Base classes for fake backends
from .fake_backend import FakeBackend
from .fake_qasm_backend import FakeQasmBackend
from .fake_pulse_backend import FakePulseBackend

# Fake providers
from .fake_provider import FakeProviderFactory, FakeProviderForBackendV2, FakeProvider

# Standard fake backends with IBM Quantum systems snapshots
from .backends import *

# Special fake backends for special testing perpurposes
from .fake_qasm_simulator import FakeQasmSimulator
from .fake_openpulse_2q import FakeOpenPulse2Q
from .fake_openpulse_3q import FakeOpenPulse3Q
from .fake_1q import Fake1Q
from .fake_backend_v2 import FakeBackendV2, FakeBackend5QV2
from .fake_mumbai_v2 import FakeMumbaiFractionalCX

# Configurable fake backend
from .utils.configurable_backend import ConfigurableFakeBackend
