# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests PassManagerConfig"""

from qiskit import QuantumRegister
from qiskit.providers.backend import Backend
from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeMelbourne
from qiskit.test.mock.backends.almaden.fake_almaden import FakeAlmaden
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.passmanager_config import PassManagerConfig


class TestPassManagerConfig(QiskitTestCase):
    """Test PassManagerConfig.from_backend()."""

    def test_config_from_backend(self):
        """Test from_backend() with a valid backend.

        `FakeAlmaden` is used in this testcase. This backend has `defaults` attribute
        that contains an instruction schedule map.
        """
        backend = FakeAlmaden()
        config = PassManagerConfig.from_backend(backend)
        self.assertEqual(config.basis_gates, backend.configuration().basis_gates)
        self.assertEqual(config.inst_map, backend.defaults().instruction_schedule_map)
        self.assertEqual(
            str(config.coupling_map), str(CouplingMap(backend.configuration().coupling_map))
        )

    def test_invalid_backend(self):
        """Test from_backend() with an invalid backend."""
        with self.assertRaises(AttributeError):
            PassManagerConfig.from_backend(Backend())

    def test_from_backend_and_user(self):
        """Test from_backend() with a backend and user options.

        `FakeMelbourne` is used in this testcase. This backend does not have
        `defaults` attribute and thus not provide an instruction schedule map.
        """
        qr = QuantumRegister(4, "qr")
        initial_layout = [None, qr[0], qr[1], qr[2], None, qr[3]]

        backend = FakeMelbourne()
        config = PassManagerConfig.from_backend(
            backend, basis_gates=["user_gate"], initial_layout=initial_layout
        )
        self.assertEqual(config.basis_gates, ["user_gate"])
        self.assertNotEqual(config.basis_gates, backend.configuration().basis_gates)
        self.assertIsNone(config.inst_map)
        self.assertEqual(
            str(config.coupling_map), str(CouplingMap(backend.configuration().coupling_map))
        )
        self.assertEqual(config.initial_layout, initial_layout)

    def test_invalid_user_option(self):
        """Test from_backend() with an invalid user option."""
        with self.assertRaises(TypeError):
            PassManagerConfig.from_backend(FakeMelbourne(), invalid_option=None)
