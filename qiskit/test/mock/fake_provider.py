# -*- coding: utf-8 -*-

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
Fake provider class that provides access to fake backends.
"""

from qiskit.providers.baseprovider import BaseProvider
from qiskit.providers.exceptions import QiskitBackendNotFoundError

from .fake_qasm_simulator import FakeQasmSimulator
from .fake_yorktown import FakeYorktown
from .fake_tenerife import FakeTenerife
from .fake_ourense import FakeOurense
from .fake_melbourne import FakeMelbourne
from .fake_rueschlikon import FakeRueschlikon
from .fake_tokyo import FakeTokyo
from .fake_poughkeepsie import FakePoughkeepsie
from .fake_boeblingen import FakeBoeblingen
from .fake_openpulse_2q import FakeOpenPulse2Q


class FakeProvider(BaseProvider):
    """Dummy provider just for testing purposes.

    Only filtering backends by name is implemented.
    """

    def get_backend(self, name=None, **kwargs):
        backend = self._backends[0]
        if name:
            filtered_backends = [backend for backend in self._backends
                                 if backend.name() == name]
            if not filtered_backends:
                raise QiskitBackendNotFoundError()

            backend = filtered_backends[0]

        return backend

    def backends(self, name=None, **kwargs):
        return self._backends

    def __init__(self):
        self._backends = [FakeQasmSimulator(),
                          FakeOpenPulse2Q(),
                          FakeYorktown(),
                          FakeTenerife(),
                          FakeOurense(),
                          FakeMelbourne(),
                          FakeRueschlikon(),
                          FakeTokyo(),
                          FakePoughkeepsie(),
                          FakeBoeblingen()]

        super().__init__()
