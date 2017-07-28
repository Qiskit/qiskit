import os

def load_tests(loader, standard_tests, pattern):
    """
    test suite for unittest discovery
    """
    this_dir = os.path.dirname(__file__)
    if pattern in ['test*.py', '*_test.py']:
        package_tests = loader.discover(start_dir=this_dir, pattern=pattern)
        standard_tests.addTests(package_tests)
    elif pattern in ['profile*.py', '*_profile.py']:
        loader.testMethodPrefix = 'profile'
        package_tests = loader.discover(start_dir=this_dir, pattern='test*.py')
        standard_tests.addTests(package_tests)
    return standard_tests
