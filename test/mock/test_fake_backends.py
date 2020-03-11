# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test of generated fake backends."""

from qiskit.test import QiskitTestCase
from qiskit.test.mock.utils.fake_backend_builder import FakeBackendBuilder


class GeneratedFakeBackendsTest(QiskitTestCase):
    """Generated fake backends test."""

    def setUp(self) -> None:
        FakeBackendClass = FakeBackendBuilder("FakeTashkent", n_qubits=4).build()
        self.backend = FakeBackendClass()

    def test_not_even_came_up_with_name_yet(self):
        # TODO: rename and implement something like scheduling circuit
        pass
