from ._qasmsimulator import QasmSimulator
from ._unitarysimulator import UnitarySimulator

def load_tests(loader, standard_tests, pattern):
    """
    test suite for unittest discovery
    """
    from . import _qasmsimulator_test
    if pattern in ['test*.py', '*_test.py']:
        suite = _qasmsimulator_test.generateTestSuite()
    elif pattern in ['*_profile.py']:
        suite = _qasmsimulator_test.generateProfileSuite()
    else:
        suite = None
    return suite
