import unittest
import os
from . import test_quantumprogram
from . import test_qasm_python_simulator
from . import test_unitary_python_simulator

def load_tests(loader, standard_tests, pattern):
    """
    test suite for unittest discovery
    """
    suiteList = []
    # should probably catch profile pattern in load_tests() of individual modules.
    if pattern in ['test*.py', '*_test.py']:
        suiteList.append(unittest.defaultTestLoader.loadTestsFromModule(test_quantumprogram))
        suiteList.append(test_qasm_python_simulator.generateTestSuite())
        suiteList.append(test_unitary_python_simulator.generateTestSuite())        
    elif pattern in ['profile*.py', '*_profile.py']:
        suiteList.append(test_qasm_python_simulator.generateProfileSuite())
        suiteList.append(test_unitary_python_simulator.generateProfileSuite())        
    return unittest.TestSuite(suiteList)
