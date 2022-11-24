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

# pylint: disable=missing-class-docstring,missing-function-docstring
# pylint: disable=missing-module-docstring

from ddt import ddt

from qiskit.test.base import QiskitTestCase
from qiskit.providers.fake_provider import (
    FakeProviderForBackendV2,
    FakeBackendV2,
    FakeProvider,
    FakePulseBackend,
    FakeQasmBackend,
)


@ddt
class TestFakeProviderForBackendV2(QiskitTestCase):
    def setUp(self):
        super().setUp()
        self.provider = FakeProviderForBackendV2()

    def test_get_backend(self):
        backend_name = "fake_manila_v2"
        backend = self.provider.get_backend(backend_name)
        self.assertTrue(isinstance(backend, FakeBackendV2))

    def test_backends(self):
        backends = self.provider.backends()
        self.assertTrue(isinstance(backends, list))


@ddt
class TestFakeProvider(QiskitTestCase):
    def setUp(self):
        super().setUp()
        self.provider = FakeProvider()

    def test_get_backend(self):
        backend_name = "fake_manila"
        backend = self.provider.get_backend(backend_name)
        self.assertTrue(isinstance(backend, (FakePulseBackend, FakeQasmBackend)))

    def test_backends(self):
        backends = self.provider.backends()
        self.assertTrue(isinstance(backends, list))
