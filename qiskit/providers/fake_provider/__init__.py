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

The fake provider module contains fake providers and fake backends classes. The fake backends are
built to mimic the behaviors of IBM Quantum systems using system snapshots. The system snapshots
contain important information about the quantum system such as coupling map, basis gates, qubit
properties (T1, T2, error rate, etc.) which are useful for testing the transpiler and performing
noisy simulation of the system.

Example Usage
=============

Here is an example of using a fake backend for transpilation and simulation.

.. plot::
   :include-source:

   from qiskit import QuantumCircuit
   from qiskit.providers.fake_provider import FakeManilaV2
   from qiskit import transpile
   from qiskit.tools.visualization import plot_histogram


   # Get a fake backend from the fake provider
   backend = FakeManilaV2()

   # Create a simple circuit
   circuit = QuantumCircuit(3)
   circuit.h(0)
   circuit.cx(0,1)
   circuit.cx(0,2)
   circuit.measure_all()
   circuit.draw('mpl')

   # Transpile the ideal circuit to a circuit that can be directly executed by the backend
   transpiled_circuit = transpile(circuit, backend)
   transpiled_circuit.draw('mpl')

   # Run the transpiled circuit using the simulated fake backend
   job = backend.run(transpiled_circuit)
   counts = job.result().get_counts()
   plot_histogram(counts)

.. important::

    Please note that the simulation is done using a noise model generated from system snapshots
    obtained in the past (sometimes a few years ago) and the results are not representative of the
    latest behaviours of the real quantum system which the fake backend is mimicking. If you want to
    run noisy simulations to compare with the real quantum system, please follow steps below to
    generate a simulator mimics a real quantum system with the latest calibration results.

    .. code-block:: python

        from qiskit.providers.ibmq import IBMQ
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

    FakeProviderForBackendV2
    FakeProvider

Fake Backends
=============

Fake V2 Backends
----------------

Fake V2 backends are fake backends with IBM Quantum systems snapshots implemented with
:mod:`~qiskit.providers.backend.BackendV2` interface.  They are all subclasses of
:class:`FakeBackendV2`.

.. autosummary::
    :toctree: ../stubs/

    FakeAlmadenV2
    FakeArmonkV2
    FakeAthensV2
    FakeAuckland
    FakeBelemV2
    FakeBoeblingenV2
    FakeBogotaV2
    FakeBrooklynV2
    FakeBurlingtonV2
    FakeCairoV2
    FakeCambridgeV2
    FakeCasablancaV2
    FakeEssexV2
    FakeGeneva
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
    FakeOslo
    FakeOurenseV2
    FakeParisV2
    FakePerth
    FakePrague
    FakePoughkeepsieV2
    FakeQuitoV2
    FakeRochesterV2
    FakeRomeV2
    .. FakeRueschlikonV2 # no v2 version
    FakeSantiagoV2
    FakeSherbrooke
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

Fake Backend Base Classes
=========================

The fake backends based on IBM hardware are based on a set of base classes:

.. currentmodule:: qiskit.providers.fake_provider.fake_backend
.. autoclass:: qiskit.providers.fake_provider.fake_backend.FakeBackendV2

.. currentmodule:: qiskit.providers.fake_provider
.. autoclass:: FakeBackend
.. autoclass:: FakeQasmBackend
.. autoclass:: FakePulseBackend
"""

# Fake job and qobj classes
from .fake_job import FakeJob
from .fake_qobj import FakeQobj

# Base classes for fake backends
from . import fake_backend
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
