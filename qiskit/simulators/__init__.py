import unittest
from ._qasmsimulator import QasmSimulator
from ._unitarysimulator import UnitarySimulator

def load_tests(loader, standard_tests, pattern):
    """
    test suite for unittest discovery
    """
    from . import _qasmsimulator_test
    suiteList = []
    if pattern in ['test*.py', '*_test.py']:
        suiteList.append(_qasmsimulator_test.generateTestSuite())
    elif pattern in ['*_profile.py']:
        suiteList.append(_qasmsimulator_test.generateProfileSuite())
    else:
        suiteList.append(None)
    return unittest.TestSuite(suiteList)
