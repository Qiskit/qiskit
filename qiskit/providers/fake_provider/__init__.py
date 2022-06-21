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

.. jupyter-execute::

    from qiskit import QuantumCircuit, transpile
    from qiskit.providers.fake_provider import FakeProviderForBackendV2

    # get a fake backend from a fake provider
    provider = FakeProviderForBackendV2()
    backend = provider.get_backend('fake_manila_v2')

    # create a simple circuit
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0,1)
    qc.measure_all()

    # transpile the circuit and run using the simulated fake backend
    tqc = transpile(qc, backend=backend)
    job = backend.run(tqc)
    print(job.result().get_counts())

Please note that the simulation is done using the snapshots obtained in the past (sometimes a few
years ago) and probably will not represent the latest behaviour of the real quantum system which the
fake backend is mimicking. If you want to run noisy simulations to compare with the real quantum
system, please follow steps below to generate a simulator mimics a real quantum system with the
latest calibration results.

.. code-block:: python

    from qiskit import IBMQ
    from qiskit.providers.aer import AerSimulator

    # get a real backend from a real provider
    provider = IBMQ.load_account()
    backend = provider.get_backend('ibmq_manila')

    # generate a simulator that mimics the real quantum system with the latest calibration results
    backend_sim = AerSimulator.from_backend(backend)


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
