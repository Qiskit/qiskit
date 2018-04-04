# -*- coding: utf-8 -*-
# pylint: disable=invalid-name,missing-docstring

# Copyright 2017 IBM RESEARCH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import unittest
import qiskit
from qiskit import QISKitError
from qiskit.backends import BaseBackend
from qiskit._backend_manager import (register_backend, local_backends, remote_backends,
                                     discover_backend_classes, _REGISTERED_BACKENDS)
from .common import QiskitTestCase


class TestBaseBackend(QiskitTestCase):
    def setUp(self):
        # pylint: disable=redefined-outer-name,unused-variable
        QiskitTestCase.setUp(self)

        # Manually clear and populate the list of registered backends, as it is
        # defined at module scope and computed during the initial import.
        _REGISTERED_BACKENDS = {}
        discover_backend_classes(qiskit)

    def test_register_valid_class(self):
        """Test backend registration for a custom valid backend."""
        backend_name = register_backend(ValidBackend)

        # Check that it has been added to the list of backends.
        self.assertIn(backend_name, _REGISTERED_BACKENDS.keys())
        self.assertEqual(_REGISTERED_BACKENDS[backend_name].cls, ValidBackend)

        # Second registration should fail as it is already registered.
        with self.assertRaises(QISKitError):
            register_backend(ValidBackend)

    def test_register_invalid_class(self):
        """Test backend registration for invalid backends."""
        initial_backends_len = len(_REGISTERED_BACKENDS)

        for backend_cls in [NotSubclassBackend,
                            NoConfigurationBackend,
                            UninstantiableBackend]:
            with self.subTest(backend_cls=backend_cls):
                with self.assertRaises(QISKitError):
                    register_backend(backend_cls)

        self.assertEqual(initial_backends_len, len(_REGISTERED_BACKENDS))

    def test_fail_incomplete_implementation(self):
        # pylint: disable=abstract-method,super-init-not-called,abstract-class-instantiated
        class SubclassTest(BaseBackend):
            def __init__(self, qobj):
                pass

            @property
            def configuration(self):
                pass

        self.assertTrue(issubclass(SubclassTest, BaseBackend))
        with self.assertRaises(TypeError) as type_err:
            _ = SubclassTest({})
        exc = type_err.exception
        self.assertEqual(str(exc), ("Can't instantiate abstract class "
                                    "SubclassTest with abstract methods run"))

    def test_local_backends(self):
        available_backends = local_backends()
        self.log.info('The discovered local devices are: %s', available_backends)

        # Some local backends should always be present.
        self.assertIn('local_qasm_simulator', available_backends)
        self.assertIn('local_unitary_simulator', available_backends)

    def test_remote_backends(self):
        self.log.info('The discovered remote devices are: %s', remote_backends())


# Dummy backend classes for testing registration.
class NoConfigurationBackend(BaseBackend):
    def __init__(self, configuration=None):
        # pylint: disable=super-init-not-called
        pass

    def run(self, q_job):
        pass

    @property
    def configuration(self):
        pass


class ValidBackend(NoConfigurationBackend):
    def __init__(self, configuration=None):
        # pylint: disable=super-init-not-called
        if not configuration:
            self._configuration = {'name': 'valid_backend', 'local': True}
        else:
            self._configuration = configuration

    @property
    def configuration(self):
        return self._configuration


class UninstantiableBackend(ValidBackend):
    def __init__(self):
        # pylint: disable=super-init-not-called
        raise Exception


class NotSubclassBackend(object):
    pass


if __name__ == '__main__':
    unittest.main(verbosity=2)
