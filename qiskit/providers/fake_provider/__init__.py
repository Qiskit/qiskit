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

The fake provider module contains fake providers and fake backends classes, primarily used for unit
testing of the transpiler.

Fake Providers
==============

Fake providers provide access to a list of fake backends.

.. autosummary::
    :toctree: ../stubs/

    FakeProviderFactory
    FakeProviderForBackendV2
    FakeProvider

Fake Backends
=============

Fake V2 Backends
----------------

Fake V2 backends are fake backends with IBM Quantum systems snapshots implemented with
:mod:`~qiskit.providers.backend.BackendV2` interface.

.. autosummary::
    :toctree: ../stubs/

    FakeAlmadenV2
    FakeArmonkV2
    FakeAthensV2
    FakeBelemV2
    FakeBoeblingenV2
    FakeBogotaV2
    FakeBrooklynV2
    FakeBurlingtonV2
    FakeCairoV2
    FakeCambridgeV2
    FakeCasablancaV2
    FakeEssexV2
    FakeGuadalupeV2
    FakeHanoiV2
    FakeJakartaV2
    FakeJohannesburgV2
    FakeKolkataV2
    FakeLagosV2
    FakeLimaV2
    FakeLondonV2
    FakeManhattanV2
    FakeManilaV2
    FakeMelbourneV2
    FakeMontrealV2
    FakeMumbaiV2
    FakeNairobiV2
    FakeOurenseV2
    FakeParisV2
    FakePoughkeepsieV2
    FakeQuitoV2
    FakeRochesterV2
    FakeRomeV2
    .. FakeRueschlikonV2 # no v2 version
    FakeSantiagoV2
    FakeSingaporeV2
    FakeSydneyV2
    .. FakeTenerifeV2 # no v2 version
    .. FakeTokyoV2 # no v2 version
    FakeTorontoV2
    FakeValenciaV2
    FakeVigoV2
    FakeWashingtonV2
    FakeYorktownV2

Fake V1 Backends
----------------

Fake V1 backends are fake backends with IBM Quantum systems snapshots implemented with
:mod:`~qiskit.providers.backend.BackendV1` interface.

.. autosummary::
    :toctree: ../stubs/

    FakeAlmaden
    FakeArmonk
    FakeAthens
    FakeBelem
    FakeBoeblingen
    FakeBogota
    FakeBrooklyn
    FakeBurlington
    FakeCairo
    FakeCambridge
    FakeCasablanca
    FakeEssex
    FakeGuadalupe
    FakeHanoi
    FakeJakarta
    FakeJohannesburg
    FakeKolkata
    FakeLagos
    FakeLima
    FakeLondon
    FakeManhattan
    FakeManila
    FakeMelbourne
    FakeMontreal
    FakeMumbai
    FakeNairobi
    FakeOurense
    FakeParis
    FakePoughkeepsie
    FakeQuito
    FakeRochester
    FakeRome
    FakeRueschlikon
    FakeSantiago
    FakeSingapore
    FakeSydney
    FakeTenerife
    FakeTokyo
    FakeToronto
    FakeValencia
    FakeVigo
    FakeWashington
    FakeYorktown

Special Fake Backends
=====================

Special fake backends are fake backends that were created for special testing purposes.

.. autosummary::
    :toctree: ../stubs/

    FakeQasmSimulator
    FakeOpenPulse2Q
    FakeOpenPulse3Q
    Fake1Q
    FakeBackendV2
    FakeBackend5QV2
    FakeMumbaiFractionalCX
    ConfigurableFakeBackend

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
