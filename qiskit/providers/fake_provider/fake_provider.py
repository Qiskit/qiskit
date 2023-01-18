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

# pylint: disable=wildcard-import,unused-argument

"""
Fake provider class that provides access to fake backends.
"""

from qiskit.providers.provider import ProviderV1
from qiskit.providers.exceptions import QiskitBackendNotFoundError

from .backends import *
from .fake_qasm_simulator import FakeQasmSimulator
from .fake_openpulse_2q import FakeOpenPulse2Q
from .fake_openpulse_3q import FakeOpenPulse3Q


class FakeProviderFactory:
    """Fake provider factory class."""

    def __init__(self):
        self.fake_provider = FakeProvider()

    def load_account(self):
        """Fake load_account method to mirror the IBMQ provider."""
        pass

    def enable_account(self, *args, **kwargs):
        """Fake enable_account method to mirror the IBMQ provider factory."""
        pass

    def disable_account(self):
        """Fake disable_account method to mirror the IBMQ provider factory."""
        pass

    def save_account(self, *args, **kwargs):
        """Fake save_account method to mirror the IBMQ provider factory."""
        pass

    @staticmethod
    def delete_account():
        """Fake delete_account method to mirror the IBMQ provider factory."""
        pass

    def update_account(self, force=False):
        """Fake update_account method to mirror the IBMQ provider factory."""
        pass

    def providers(self):
        """Fake providers method to mirror the IBMQ provider."""
        return [self.fake_provider]

    def get_provider(self, hub=None, group=None, project=None):
        """Fake get_provider method to mirror the IBMQ provider."""
        return self.fake_provider


class FakeProviderForBackendV2(ProviderV1):
    """Fake provider containing fake V2 backends.

    Only filtering backends by name is implemented. This class contains all fake V2 backends
    available in the :mod:`qiskit.providers.fake_provider`.
    """

    def get_backend(self, name=None, **kwargs):
        backend = self._backends[0]
        if name:
            filtered_backends = [backend for backend in self._backends if backend.name() == name]
            if not filtered_backends:
                raise QiskitBackendNotFoundError()

            backend = filtered_backends[0]

        return backend

    def backends(self, name=None, **kwargs):
        return self._backends

    def __init__(self):
        self._backends = [
            FakeAlmadenV2(),
            FakeArmonkV2(),
            FakeAthensV2(),
            FakeAuckland(),
            FakeBelemV2(),
            FakeBoeblingenV2(),
            FakeBogotaV2(),
            FakeBrooklynV2(),
            FakeBurlingtonV2(),
            FakeCairoV2(),
            FakeCambridgeV2(),
            FakeCasablancaV2(),
            FakeEssexV2(),
            FakeGeneva(),
            FakeGuadalupeV2(),
            FakeHanoiV2(),
            FakeJakartaV2(),
            FakeJohannesburgV2(),
            FakeKolkataV2(),
            FakeLagosV2(),
            FakeLimaV2(),
            FakeLondonV2(),
            FakeManhattanV2(),
            FakeManilaV2(),
            FakeMelbourneV2(),
            FakeMontrealV2(),
            FakeMumbaiV2(),
            FakeNairobiV2(),
            FakeOslo(),
            FakeOurenseV2(),
            FakeParisV2(),
            FakePerth(),
            FakePrague(),
            FakePoughkeepsieV2(),
            FakeQuitoV2(),
            FakeRochesterV2(),
            FakeRomeV2(),
            FakeSantiagoV2(),
            FakeSherbrooke(),
            FakeSingaporeV2(),
            FakeSydneyV2(),
            FakeTorontoV2(),
            FakeValenciaV2(),
            FakeVigoV2(),
            FakeWashingtonV2(),
            FakeYorktownV2(),
        ]

        super().__init__()


class FakeProvider(ProviderV1):
    """Fake provider containing fake V1 backends.

    Only filtering backends by name is implemented. This class contains all fake V1 backends
    available in the :mod:`qiskit.providers.fake_provider`.
    """

    def get_backend(self, name=None, **kwargs):
        backend = self._backends[0]
        if name:
            filtered_backends = [backend for backend in self._backends if backend.name() == name]
            if not filtered_backends:
                raise QiskitBackendNotFoundError()

            backend = filtered_backends[0]

        return backend

    def backends(self, name=None, **kwargs):
        return self._backends

    def __init__(self):
        self._backends = [
            FakeAlmaden(),
            FakeArmonk(),
            FakeAthens(),
            FakeBelem(),
            FakeBoeblingen(),
            FakeBogota(),
            FakeBrooklyn(),
            FakeBurlington(),
            FakeCairo(),
            FakeCambridge(),
            FakeCambridgeAlternativeBasis(),
            FakeCasablanca(),
            FakeEssex(),
            FakeGuadalupe(),
            FakeHanoi(),
            FakeJakarta(),
            FakeJohannesburg(),
            FakeKolkata(),
            FakeLagos(),
            FakeLima(),
            FakeLondon(),
            FakeManila(),
            FakeManhattan(),
            FakeMelbourne(),
            FakeMontreal(),
            FakeMumbai(),
            FakeNairobi(),
            FakeOpenPulse2Q(),
            FakeOpenPulse3Q(),
            FakeOurense(),
            FakeParis(),
            FakePoughkeepsie(),
            FakeQasmSimulator(),
            FakeQuito(),
            FakeRochester(),
            FakeRome(),
            FakeRueschlikon(),
            FakeSantiago(),
            FakeSingapore(),
            FakeSydney(),
            FakeTenerife(),
            FakeTokyo(),
            FakeToronto(),
            FakeValencia(),
            FakeVigo(),
            FakeWashington(),
            FakeYorktown(),
        ]

        super().__init__()
