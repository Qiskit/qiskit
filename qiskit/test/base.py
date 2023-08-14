# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name

"""Base TestCases for the unit tests.

Implementors of unit tests for Terra are encouraged to subclass
``QiskitTestCase`` in order to take advantage of utility functions (for example,
the environment variables for customizing different options), and the
decorators in the ``decorators`` package.
"""

import inspect
import logging
import os
import sys
import warnings
import unittest
from unittest.util import safe_repr

from qiskit.tools.parallel import get_platform_parallel_default
from qiskit.utils import optionals as _optionals
from qiskit.circuit import QuantumCircuit
from .decorators import enforce_subclasses_call
from .utils import Path, setup_test_logging


__unittest = True  # Allows shorter stack trace for .assertDictAlmostEqual


# If testtools is installed use that as a (mostly) drop in replacement for
# unittest's TestCase. This will enable the fixtures used for capturing stdout
# stderr, and pylogging to attach the output to stestr's result stream.
if _optionals.HAS_TESTTOOLS:
    import testtools

    class BaseTestCase(testtools.TestCase):
        """Base test class."""

        # testtools maintains their own version of assert functions which mostly
        # behave as value adds to the std unittest assertion methods. However,
        # for assertEquals and assertRaises modern unittest has diverged from
        # the forks in testtools and offer more (or different) options that are
        # incompatible testtools versions. Just use the stdlib versions so that
        # our tests work as expected.
        assertRaises = unittest.TestCase.assertRaises
        assertEqual = unittest.TestCase.assertEqual

else:

    class BaseTestCase(unittest.TestCase):
        """Base test class."""

        pass


@enforce_subclasses_call(["setUp", "setUpClass", "tearDown", "tearDownClass"])
class BaseQiskitTestCase(BaseTestCase):
    """Additions for test cases for all Qiskit-family packages.

    The additions here are intended for all packages, not just Terra.  Terra-specific logic should
    be in the Terra-specific classes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__setup_called = False
        self.__teardown_called = False

    def setUp(self):
        super().setUp()
        self.addTypeEqualityFunc(QuantumCircuit, self.assertQuantumCircuitEqual)
        if self.__setup_called:
            raise ValueError(
                "In File: %s\n"
                "TestCase.setUp was already called. Do not explicitly call "
                "setUp from your tests. In your own setUp, use super to call "
                "the base setUp." % (sys.modules[self.__class__.__module__].__file__,)
            )
        self.__setup_called = True

    def tearDown(self):
        super().tearDown()
        if self.__teardown_called:
            raise ValueError(
                "In File: %s\n"
                "TestCase.tearDown was already called. Do not explicitly call "
                "tearDown from your tests. In your own tearDown, use super to "
                "call the base tearDown." % (sys.modules[self.__class__.__module__].__file__,)
            )
        self.__teardown_called = True

    @staticmethod
    def _get_resource_path(filename, path=Path.TEST):
        """Get the absolute path to a resource.

        Args:
            filename (string): filename or relative path to the resource.
            path (Path): path used as relative to the filename.

        Returns:
            str: the absolute path to the resource.
        """
        return os.path.normpath(os.path.join(path.value, filename))

    def assertQuantumCircuitEqual(self, qc1, qc2, msg=None):
        """Extra assertion method to give a better error message when two circuits are unequal."""
        if qc1 == qc2:
            return
        if msg is None:
            msg = "The two circuits are not equal."
        msg += f"""
Left circuit:
{qc1}

Right circuit:
{qc2}"""
        raise self.failureException(msg)

    def assertDictAlmostEqual(
        self, dict1, dict2, delta=None, msg=None, places=None, default_value=0
    ):
        """Assert two dictionaries with numeric values are almost equal.

        Fail if the two dictionaries are unequal as determined by
        comparing that the difference between values with the same key are
        not greater than delta (default 1e-8), or that difference rounded
        to the given number of decimal places is not zero. If a key in one
        dictionary is not in the other the default_value keyword argument
        will be used for the missing value (default 0). If the two objects
        compare equal then they will automatically compare almost equal.

        Args:
            dict1 (dict): a dictionary.
            dict2 (dict): a dictionary.
            delta (number): threshold for comparison (defaults to 1e-8).
            msg (str): return a custom message on failure.
            places (int): number of decimal places for comparison.
            default_value (number): default value for missing keys.

        Raises:
            TypeError: if the arguments are not valid (both `delta` and
                `places` are specified).
            AssertionError: if the dictionaries are not almost equal.
        """

        error_msg = dicts_almost_equal(dict1, dict2, delta, places, default_value)

        if error_msg:
            msg = self._formatMessage(msg, error_msg)
            raise self.failureException(msg)

    def enable_parallel_processing(self):
        """
        Enables parallel processing, for the duration of a test, on platforms
        that support it. This is done by temporarily overriding the value of
        the QISKIT_PARALLEL environment variable with the platform specific default.
        """
        parallel_default = str(get_platform_parallel_default()).upper()

        def set_parallel_env(name, value):
            os.environ[name] = value

        self.addCleanup(
            lambda value: set_parallel_env("QISKIT_PARALLEL", value),
            os.getenv("QISKIT_PARALLEL", parallel_default),
        )

        os.environ["QISKIT_PARALLEL"] = parallel_default


class QiskitTestCase(BaseQiskitTestCase):
    """Terra-specific extra functionality for test cases."""

    def tearDown(self):
        super().tearDown()
        # Reset the default providers, as in practice they acts as a singleton
        # due to importing the instances from the top-level qiskit namespace.
        from qiskit.providers.basicaer import BasicAer

        BasicAer._backends = BasicAer._verify_backends()

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Determines if the TestCase is using IBMQ credentials.
        cls.using_ibmq_credentials = False
        # Set logging to file and stdout if the LOG_LEVEL envar is set.
        cls.log = logging.getLogger(cls.__name__)
        if os.getenv("LOG_LEVEL"):
            filename = "%s.log" % os.path.splitext(inspect.getfile(cls))[0]
            setup_test_logging(cls.log, os.getenv("LOG_LEVEL"), filename)

        warnings.filterwarnings("error", category=DeprecationWarning)
        allow_DeprecationWarning_modules = [
            "test.python.pulse.test_parameters",
            "test.python.pulse.test_transforms",
            "test.python.circuit.test_gate_power",
            "test.python.pulse.test_builder",
            "test.python.pulse.test_block",
            "test.python.quantum_info.operators.symplectic.test_legacy_pauli",
            "qiskit.quantum_info.operators.pauli",
            "pybobyqa",
            "numba",
            "qiskit.utils.measurement_error_mitigation",
            "qiskit.circuit.library.standard_gates.x",
            "qiskit.pulse.schedule",
            "qiskit.pulse.instructions.instruction",
            "qiskit.pulse.instructions.play",
            "qiskit.pulse.library.parametric_pulses",
            "qiskit.quantum_info.operators.symplectic.pauli",
            "test.python.dagcircuit.test_dagcircuit",
            "importlib_metadata",
        ]
        for mod in allow_DeprecationWarning_modules:
            warnings.filterwarnings("default", category=DeprecationWarning, module=mod)
        allow_DeprecationWarning_message = [
            r"elementwise comparison failed.*",
            r"The jsonschema validation included in qiskit-terra.*",
            r"The DerivativeBase.parameter_expression_grad method.*",
            r"The property ``qiskit\.circuit\.bit\.Bit\.(register|index)`` is deprecated.*",
            r"The CXDirection pass has been deprecated",
            r"The pauli_basis function with PauliTable.*",
            # Caused by internal scikit-learn scipy usage
            r"The 'sym_pos' keyword is deprecated and should be replaced by using",
            # jupyter_client 7.4.8 uses deprecated shims in pyzmq that raise warnings with pyzmq 25.
            # These are due to be fixed by jupyter_client 8, see:
            #   - https://github.com/jupyter/jupyter_client/issues/913
            #   - https://github.com/jupyter/jupyter_client/pull/842
            r"zmq\.eventloop\.ioloop is deprecated in pyzmq .*",
        ]
        for msg in allow_DeprecationWarning_message:
            warnings.filterwarnings("default", category=DeprecationWarning, message=msg)
        # This warning should be fixed once Qiskit/qiskit-aer#1761 is in a release version of Aer.
        warnings.filterwarnings(
            "default",
            category=DeprecationWarning,
            module="qiskit_aer.*",
            message="Setting metadata to None.*",
        )


class FullQiskitTestCase(QiskitTestCase):
    """Terra-specific further additions for test cases, if ``testtools`` is available.

    It is not normally safe to derive from this class by name; on import, Terra checks if the
    necessary packages are available, and binds this class to the name :obj:`~QiskitTestCase` if so.
    If you derive directly from it, you may try and instantiate the class without satisfying its
    dependencies."""

    @_optionals.HAS_FIXTURES.require_in_call("output-capturing test cases")
    def setUp(self):
        import fixtures

        super().setUp()
        if os.environ.get("QISKIT_TEST_CAPTURE_STREAMS"):
            stdout = self.useFixture(fixtures.StringStream("stdout")).stream
            self.useFixture(fixtures.MonkeyPatch("sys.stdout", stdout))
            stderr = self.useFixture(fixtures.StringStream("stderr")).stream
            self.useFixture(fixtures.MonkeyPatch("sys.stderr", stderr))
            self.useFixture(fixtures.LoggerFixture(nuke_handlers=False, level=None))


def dicts_almost_equal(dict1, dict2, delta=None, places=None, default_value=0):
    """Test if two dictionaries with numeric values are almost equal.

    Fail if the two dictionaries are unequal as determined by
    comparing that the difference between values with the same key are
    not greater than delta (default 1e-8), or that difference rounded
    to the given number of decimal places is not zero. If a key in one
    dictionary is not in the other the default_value keyword argument
    will be used for the missing value (default 0). If the two objects
    compare equal then they will automatically compare almost equal.

    Args:
        dict1 (dict): a dictionary.
        dict2 (dict): a dictionary.
        delta (number): threshold for comparison (defaults to 1e-8).
        places (int): number of decimal places for comparison.
        default_value (number): default value for missing keys.

    Raises:
        TypeError: if the arguments are not valid (both `delta` and
            `places` are specified).

    Returns:
        String: Empty string if dictionaries are almost equal. A description
            of their difference if they are deemed not almost equal.
    """

    def valid_comparison(value):
        """compare value to delta, within places accuracy"""
        if places is not None:
            return round(value, places) == 0
        else:
            return value < delta

    # Check arguments.
    if dict1 == dict2:
        return ""
    if places is not None:
        if delta is not None:
            raise TypeError("specify delta or places not both")
        msg_suffix = " within %s places" % places
    else:
        delta = delta or 1e-8
        msg_suffix = " within %s delta" % delta

    # Compare all keys in both dicts, populating error_msg.
    error_msg = ""
    for key in set(dict1.keys()) | set(dict2.keys()):
        val1 = dict1.get(key, default_value)
        val2 = dict2.get(key, default_value)
        if not valid_comparison(abs(val1 - val2)):
            error_msg += f"({safe_repr(key)}: {safe_repr(val1)} != {safe_repr(val2)}), "

    if error_msg:
        return error_msg[:-2] + msg_suffix
    else:
        return ""


# Maintain naming backwards compatibility for downstream packages.
BasicQiskitTestCase = QiskitTestCase

if _optionals.HAS_TESTTOOLS and _optionals.HAS_FIXTURES:
    QiskitTestCase = FullQiskitTestCase
