# pylint: disable=invalid-name,missing-docstring

import unittest

from qiskit import QISKitError
from qiskit.backends import (BaseBackend,
                             local_backends,
                             remote_backends,
                             register_backend)
from qiskit.backends._backendutils import (_REGISTERED_BACKENDS,
                                           discover_local_backends)

from .common import QiskitTestCase


class TestBaseBackend(QiskitTestCase):
    def setUp(self):
        QiskitTestCase.setUp(self)

        # Manually clear and populate the list of registered backends, as it is
        # defined at module scope and computed during the initial import.
        _REGISTERED_BACKENDS = {}
        discover_local_backends()

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

        with self.assertRaises(QISKitError):
            register_backend(NotSubclassBackend)

        with self.assertRaises(QISKitError):
            register_backend(NoConfigurationBackend)

        with self.assertRaises(QISKitError):
            register_backend(UninstantiableBackend)

        self.assertEqual(initial_backends_len, len(_REGISTERED_BACKENDS))

    def test_fail_incomplete_implementation(self):
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
        self.log.info('The discovered local devices are: {}'.format(
            available_backends))

        # Some local backends should always be present.
        self.assertIn('local_qasm_simulator', available_backends)
        self.assertIn('local_unitary_simulator', available_backends)

    def test_remote_backends(self):
        self.log.info('The discovered remote devices are: {}'.format(
            remote_backends()))


# Dummy backend classes for testing registration.
class NoConfigurationBackend(BaseBackend):
    def __init__(self, configuration=None):
        pass

    def run(self, q_job):
        pass

    @property
    def configuration(self):
        pass


class ValidBackend(NoConfigurationBackend):
    def __init__(self, configuration=None):
        if configuration == None:
            self._configuration = {'name':'valid_backend'}
        else:
            self._configuration = configuration

    @property
    def configuration(self):
        return self._configuration


class UninstantiableBackend(ValidBackend):
    def __init__(self):
        raise Exception


class NotSubclassBackend(object):
    pass


if __name__ == '__main__':
    unittest.main(verbosity=2)
