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

# pylint: disable=attribute-defined-outside-init,invalid-name,missing-type-doc
# pylint: disable=unused-argument,broad-except,bad-staticmethod-argument
# pylint: disable=inconsistent-return-statements

"""Base TestCases for the unit tests.

Implementors of unit tests for Terra are encouraged to subclass
``QiskitTestCase`` in order to take advantage of utility functions (for example,
the environment variables for customizing different options), and the
decorators in the ``decorators`` package.
"""

import inspect
import itertools
import logging
import os
import sys
import warnings
import unittest
from unittest.util import safe_repr

try:
    import fixtures
    from testtools.compat import advance_iterator
    from testtools import content

    HAS_FIXTURES = True
except ImportError:
    HAS_FIXTURES = False

from qiskit.exceptions import MissingOptionalLibraryError
from .runtest import RunTest, MultipleExceptions
from .utils import Path, setup_test_logging


__unittest = True  # Allows shorter stack trace for .assertDictAlmostEqual


def _copy_content(content_object):
    """Make a copy of the given content object.

    The content within ``content_object`` is iterated and saved. This is
    useful when the source of the content is volatile, a log file in a
    temporary directory for example.

    Args:
    content_object (content.Content): A ``content.Content`` instance.

    Returns:
        content.Content: An instance with the same mime-type as
            ``content_object`` and a non-volatile copy of its content.
    """
    content_bytes = list(content_object.iter_bytes())

    def content_callback():
        return content_bytes

    return content.Content(content_object.content_type, content_callback)


def gather_details(source_dict, target_dict):
    """Merge the details from ``source_dict`` into ``target_dict``.

    ``gather_details`` evaluates all details in ``source_dict``. Do not use it
    if the details are not ready to be evaluated.

    :param source_dict: A dictionary of details will be gathered.
    :param target_dict: A dictionary into which details will be gathered.
    """
    for name, content_object in source_dict.items():
        new_name = name
        disambiguator = itertools.count(1)
        while new_name in target_dict:
            new_name = "%s-%d" % (name, advance_iterator(disambiguator))
        name = new_name
        target_dict[name] = _copy_content(content_object)


class BaseQiskitTestCase(unittest.TestCase):
    """Common extra functionality on top of unittest."""

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


class BasicQiskitTestCase(BaseQiskitTestCase):
    """Helper class that contains common functionality."""

    @classmethod
    def setUpClass(cls):
        # Determines if the TestCase is using IBMQ credentials.
        cls.using_ibmq_credentials = False

        # Set logging to file and stdout if the LOG_LEVEL envar is set.
        cls.log = logging.getLogger(cls.__name__)
        if os.getenv("LOG_LEVEL"):
            filename = "%s.log" % os.path.splitext(inspect.getfile(cls))[0]
            setup_test_logging(cls.log, os.getenv("LOG_LEVEL"), filename)

    def tearDown(self):
        # Reset the default providers, as in practice they acts as a singleton
        # due to importing the instances from the top-level qiskit namespace.
        from qiskit.providers.basicaer import BasicAer

        BasicAer._backends = BasicAer._verify_backends()


class FullQiskitTestCase(BaseQiskitTestCase):
    """Helper class that contains common functionality that captures streams."""

    run_tests_with = RunTest

    def __init__(self, *args, **kwargs):
        """Construct a TestCase."""
        if not HAS_FIXTURES:
            raise MissingOptionalLibraryError(
                libname="testtools",
                name="test runner",
                pip_install="pip install testtools",
            )
        super().__init__(*args, **kwargs)
        self.__RunTest = self.run_tests_with
        self._reset()
        self.__exception_handlers = []
        self.exception_handlers = [
            (unittest.SkipTest, self._report_skip),
            (self.failureException, self._report_failure),
            (unittest.case._UnexpectedSuccess, self._report_unexpected_success),
            (Exception, self._report_error),
        ]

    def _reset(self):
        """Reset the test case as if it had never been run."""
        self._cleanups = []
        self._unique_id_gen = itertools.count(1)
        # Generators to ensure unique traceback ids.  Maps traceback label to
        # iterators.
        self._traceback_id_gens = {}
        self.__setup_called = False
        self.__teardown_called = False
        self.__details = None

    def onException(self, exc_info, tb_label="traceback"):
        """Called when an exception propagates from test code.

        :seealso addOnException:
        """
        if exc_info[0] not in [unittest.SkipTest, unittest.case._UnexpectedSuccess]:
            self._report_traceback(exc_info, tb_label=tb_label)
        for handler in self.__exception_handlers:
            handler(exc_info)

    def _run_teardown(self, result):
        """Run the tearDown function for this test."""
        self.tearDown()
        if not self.__teardown_called:
            raise ValueError(
                "In File: %s\n"
                "TestCase.tearDown was not called. Have you upcalled all the "
                "way up the hierarchy from your tearDown? e.g. Call "
                "super(%s, self).tearDown() from your tearDown()."
                % (sys.modules[self.__class__.__module__].__file__, self.__class__.__name__)
            )

    def _get_test_method(self):
        method_name = getattr(self, "_testMethodName")
        return getattr(self, method_name)

    def _run_test_method(self, result):
        """Run the test method for this test."""
        return self._get_test_method()()

    def useFixture(self, fixture):
        """Use fixture in a test case.

        The fixture will be setUp, and self.addCleanup(fixture.cleanUp) called.

        Args:
            fixture: The fixture to use.

        Returns:
            fixture: The fixture, after setting it up and scheduling a cleanup
                for it.

        Raises:
            MultipleExceptions: When there is an error during fixture setUp
            Exception: If an exception is raised during fixture setUp
        """
        try:
            fixture.setUp()
        except MultipleExceptions as e:
            if fixtures is not None and e.args[-1][0] is fixtures.fixture.SetupError:
                gather_details(e.args[-1][1].args[0], self.getDetails())
            raise
        except Exception:
            exc_info = sys.exc_info()
            try:
                # fixture._details is not available if using the newer
                # _setUp() API in Fixtures because it already cleaned up
                # the fixture.  Ideally this whole try/except is not
                # really needed any more, however, we keep this code to
                # remain compatible with the older setUp().
                if hasattr(fixture, "_details") and fixture._details is not None:
                    gather_details(fixture.getDetails(), self.getDetails())
            except Exception:
                # Report the setUp exception, then raise the error during
                # gather_details.
                self._report_traceback(exc_info)
                raise
            else:
                # Gather_details worked, so raise the exception setUp
                # encountered.
                def reraise(exc_class, exc_obj, exc_tb, _marker=object()):
                    """Re-raise an exception received from sys.exc_info() or similar."""
                    raise exc_obj.with_traceback(exc_tb)

                reraise(*exc_info)
        else:
            self.addCleanup(fixture.cleanUp)
            self.addCleanup(gather_details, fixture.getDetails(), self.getDetails())
            return fixture

    def _run_setup(self, result):
        """Run the setUp function for this test."""
        self.setUp()
        if not self.__setup_called:
            raise ValueError(
                "In File: %s\n"
                "TestCase.setUp was not called. Have you upcalled all the "
                "way up the hierarchy from your setUp? e.g. Call "
                "super(%s, self).setUp() from your setUp()."
                % (sys.modules[self.__class__.__module__].__file__, self.__class__.__name__)
            )

    def _add_reason(self, reason):
        self.addDetail("reason", content.text_content(reason))

    @staticmethod
    def _report_error(self, result, err):
        result.addError(self, details=self.getDetails())

    @staticmethod
    def _report_expected_failure(self, result, err):
        result.addExpectedFailure(self, details=self.getDetails())

    @staticmethod
    def _report_failure(self, result, err):
        result.addFailure(self, details=self.getDetails())

    @staticmethod
    def _report_skip(self, result, err):
        if err.args:
            reason = err.args[0]
        else:
            reason = "no reason given."
        self._add_reason(reason)
        result.addSkip(self, details=self.getDetails())

    def _report_traceback(self, exc_info, tb_label="traceback"):
        id_gen = self._traceback_id_gens.setdefault(tb_label, itertools.count(0))
        while True:
            tb_id = advance_iterator(id_gen)
            if tb_id:
                tb_label = "%s-%d" % (tb_label, tb_id)
            if tb_label not in self.getDetails():
                break
        self.addDetail(
            tb_label,
            content.TracebackContent(
                exc_info, self, capture_locals=getattr(self, "__testtools_tb_locals__", False)
            ),
        )

    @staticmethod
    def _report_unexpected_success(self, result, err):
        result.addUnexpectedSuccess(self, details=self.getDetails())

    def run(self, result=None):
        self._reset()
        try:
            run_test = self.__RunTest(self, self.exception_handlers, last_resort=self._report_error)
        except TypeError:
            # Backwards compat: if we can't call the constructor
            # with last_resort, try without that.
            run_test = self.__RunTest(self, self.exception_handlers)
        return run_test.run(result)

    def setUp(self):
        super().setUp()
        if self.__setup_called:
            raise ValueError(
                "In File: %s\n"
                "TestCase.setUp was already called. Do not explicitly call "
                "setUp from your tests. In your own setUp, use super to call "
                "the base setUp." % (sys.modules[self.__class__.__module__].__file__,)
            )
        self.__setup_called = True
        if os.environ.get("QISKIT_TEST_CAPTURE_STREAMS"):
            stdout = self.useFixture(fixtures.StringStream("stdout")).stream
            self.useFixture(fixtures.MonkeyPatch("sys.stdout", stdout))
            stderr = self.useFixture(fixtures.StringStream("stderr")).stream
            self.useFixture(fixtures.MonkeyPatch("sys.stderr", stderr))
            self.useFixture(fixtures.LoggerFixture(nuke_handlers=False, level=None))

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
        # Reset the default providers, as in practice they acts as a singleton
        # due to importing the instances from the top-level qiskit namespace.
        from qiskit.providers.basicaer import BasicAer

        BasicAer._backends = BasicAer._verify_backends()

    def addDetail(self, name, content_object):
        """Add a detail to be reported with this test's outcome.

        :param name: The name to give this detail.
        :param content_object: The content object for this detail. See
            testtools.content for more detail.
        """
        if self.__details is None:
            self.__details = {}
        self.__details[name] = content_object

    def addDetailUniqueName(self, name, content_object):
        """Add a detail to the test, but ensure it's name is unique.

        This method checks whether ``name`` conflicts with a detail that has
        already been added to the test. If it does, it will modify ``name`` to
        avoid the conflict.

        :param name: The name to give this detail.
        :param content_object: The content object for this detail. See
            testtools.content for more detail.
        """
        existing_details = self.getDetails()
        full_name = name
        suffix = 1
        while full_name in existing_details:
            full_name = "%s-%d" % (name, suffix)
            suffix += 1
        self.addDetail(full_name, content_object)

    def getDetails(self):
        """Get the details dict that will be reported with this test's outcome."""
        if self.__details is None:
            self.__details = {}
        return self.__details

    @classmethod
    def setUpClass(cls):
        # Determines if the TestCase is using IBMQ credentials.
        cls.using_ibmq_credentials = False
        cls.log = logging.getLogger(cls.__name__)

        warnings.filterwarnings("error", category=DeprecationWarning)
        allow_DeprecationWarning_modules = [
            "test.ipynb.mpl.test_circuit_matplotlib_drawer",
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
            "test.python.quantum_info.operators.test_operator",
            "test.python.quantum_info.operators.test_scalar_op",
            "test.python.quantum_info.operators.test_superop",
            "test.python.quantum_info.operators.channel.test_kraus",
            "test.python.quantum_info.operators.channel.test_choi",
            "test.python.quantum_info.operators.channel.test_chi",
            "test.python.quantum_info.operators.channel.test_superop",
            "test.python.quantum_info.operators.channel.test_stinespring",
            "test.python.quantum_info.operators.symplectic.test_sparse_pauli_op",
            "test.python.quantum_info.operators.channel.test_ptm",
        ]
        for mod in allow_DeprecationWarning_modules:
            warnings.filterwarnings("default", category=DeprecationWarning, module=mod)
        allow_DeprecationWarning_message = [
            r".*LogNormalDistribution.*",
            r".*NormalDistribution.*",
            r".*UniformDistribution.*",
            r".*QuantumCircuit\.combine.*",
            r".*QuantumCircuit\.__add__.*",
            r".*QuantumCircuit\.__iadd__.*",
            r".*QuantumCircuit\.extend.*",
            r".*psi @ U.*",
            r".*qiskit\.circuit\.library\.standard_gates\.ms import.*",
            r"elementwise comparison failed.*",
            r"The jsonschema validation included in qiskit-terra.*",
            r"The DerivativeBase.parameter_expression_grad method.*",
            r"Back-references to from Bit instances.*",
            r"The QuantumCircuit.u. method.*",
            r"The QuantumCircuit.cu.",
            r"The CXDirection pass has been deprecated",
        ]
        for msg in allow_DeprecationWarning_message:
            warnings.filterwarnings("default", category=DeprecationWarning, message=msg)


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


if not HAS_FIXTURES or not os.environ.get("QISKIT_TEST_CAPTURE_STREAMS"):
    QiskitTestCase = BasicQiskitTestCase
else:
    QiskitTestCase = FullQiskitTestCase
