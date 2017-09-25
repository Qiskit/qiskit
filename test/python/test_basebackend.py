import unittest
import logging
import os
import qiskit.backends
from qiskit.backends._basebackend import BaseBackend
from qiskit.backends._backendutils import (local_backends,
                                           remote_backends)

class TestBaseBackend(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.moduleName = os.path.splitext(__file__)[0]
        cls.log = logging.getLogger(__name__)
        cls.log.setLevel(logging.INFO)
        logFileName = cls.moduleName + '.log'
        handler = logging.FileHandler(logFileName)
        handler.setLevel(logging.INFO)
        log_fmt = ('{}.%(funcName)s:%(levelname)s:%(asctime)s:'
                   ' %(message)s'.format(cls.__name__))
        formatter = logging.Formatter(log_fmt)
        handler.setFormatter(formatter)
        cls.log.addHandler(handler)

    def test_class_registration(self):
        class SubclassTest(BaseBackend):
            def __init__(self, qobj):
                pass
            def run(self):
                pass
            @property
            def configuration(self):
                pass
            @configuration.setter
            def configuration(self, conf):
                pass

        test = SubclassTest({})
        self.assertTrue(issubclass(SubclassTest, BaseBackend))
        self.assertTrue(isinstance(test, BaseBackend))

    def test_fail_incomplete_implementation(self):
        class SubclassTest(BaseBackend):
            def __init__(self, qobj):
                pass

            @property
            def configuration(self):
                pass
            @configuration.setter
            def configuration(self, conf):
                pass

        self.assertTrue(issubclass(SubclassTest, BaseBackend))
        with self.assertRaises(TypeError) as type_err:
            test = SubclassTest({})
        exc = type_err.exception
        self.assertEqual(str(exc), ("Can't instantiate abstract class "
                                    "SubclassTest with abstract methods run"))

    def test_local_backends(self):
        self.log.info('The discovered local devices are: {}'.format(
            local_backends()))

    def test_remote_backends(self):
        self.log.info('The discovered remote devices are: {}'.format(
            remote_backends()))
        
if __name__ == '__main__':
    unittest.main(verbosity=2)

