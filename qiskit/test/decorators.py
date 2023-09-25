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


"""Decorator for using with Qiskit unit tests."""

import collections.abc
import functools
import os
import socket
import sys
from typing import Union, Callable, Type, Iterable
import unittest

from qiskit.utils import wrap_method
from .testing_options import get_test_options

HAS_NET_CONNECTION = None


def _has_connection(hostname, port):
    """Checks if internet connection exists to host via specified port.

    If any exception is raised while trying to open a socket this will return
    false.

    Args:
        hostname (str): Hostname to connect to.
        port (int): Port to connect to

    Returns:
        bool: Has connection or not

    """
    try:
        host = socket.gethostbyname(hostname)
        socket.create_connection((host, port), 2).close()
        return True
    except Exception:  # pylint: disable=broad-except
        return False


def is_aer_provider_available():
    """Check if the C++ simulator can be instantiated.

    Returns:
        bool: True if simulator executable is available
    """
    # TODO: HACK FROM THE DEPTHS OF DESPAIR AS AER DOES NOT WORK ON MAC
    if sys.platform == "darwin":
        return False
    try:
        import qiskit.providers.aer  # pylint: disable=unused-import
    except ImportError:
        return False
    return True


def requires_aer_provider(test_item):
    """Decorator that skips test if qiskit aer provider is not available

    Args:
        test_item (callable): function or class to be decorated.

    Returns:
        callable: the decorated function.
    """
    reason = "Aer provider not found, skipping test"
    return unittest.skipIf(not is_aer_provider_available(), reason)(test_item)


def slow_test(func):
    """Decorator that signals that the test takes minutes to run.

    Args:
        func (callable): test function to be decorated.

    Returns:
        callable: the decorated function.
    """

    @functools.wraps(func)
    def _wrapper(*args, **kwargs):
        skip_slow = not TEST_OPTIONS["run_slow"]
        if skip_slow:
            raise unittest.SkipTest("Skipping slow tests")

        return func(*args, **kwargs)

    return _wrapper


def _get_credentials():
    """Finds the credentials for a specific test and options.

    Returns:
        Credentials: set of credentials

    Raises:
        SkipTest: when credentials can't be found
    """
    try:
        from qiskit.providers.ibmq.credentials import Credentials, discover_credentials
    except ImportError as ex:
        raise unittest.SkipTest(
            "qiskit-ibmq-provider could not be found, "
            "and is required for executing online tests. "
            'To install, run "pip install qiskit-ibmq-provider" '
            "or check your installation."
        ) from ex

    if os.getenv("IBMQ_TOKEN") and os.getenv("IBMQ_URL"):
        return Credentials(os.getenv("IBMQ_TOKEN"), os.getenv("IBMQ_URL"))
    elif os.getenv("QISKIT_TESTS_USE_CREDENTIALS_FILE"):
        # Attempt to read the standard credentials.
        discovered_credentials = discover_credentials()

        if discovered_credentials:
            # Decide which credentials to use for testing.
            if len(discovered_credentials) > 1:
                raise unittest.SkipTest(
                    "More than 1 credential set found, use: "
                    "IBMQ_TOKEN and IBMQ_URL env variables to "
                    "set credentials explicitly"
                )

            # Use the first available credentials.
            return list(discovered_credentials.values())[0]
    raise unittest.SkipTest(
        "No IBMQ credentials found for running the test. This is required for running online tests."
    )


def online_test(func):
    """Decorator that signals that the test uses the network (and the online API):

    It involves:
        * determines if the test should be skipped by checking environment
            variables.
        * if the `USE_ALTERNATE_ENV_CREDENTIALS` environment variable is
          set, it reads the credentials from an alternative set of environment
          variables.
        * if the test is not skipped, it reads `qe_token` and `qe_url` from
            `Qconfig.py`, environment variables or qiskitrc.
        * if the test is not skipped, it appends `qe_token` and `qe_url` as
            arguments to the test function.

    Args:
        func (callable): test function to be decorated.

    Returns:
        callable: the decorated function.
    """

    @functools.wraps(func)
    def _wrapper(self, *args, **kwargs):
        # To avoid checking the connection in each test
        global HAS_NET_CONNECTION  # pylint: disable=global-statement

        if TEST_OPTIONS["skip_online"]:
            raise unittest.SkipTest("Skipping online tests")

        if HAS_NET_CONNECTION is None:
            HAS_NET_CONNECTION = _has_connection("qiskit.org", 443)

        if not HAS_NET_CONNECTION:
            raise unittest.SkipTest("Test requires internet connection.")

        credentials = _get_credentials()
        self.using_ibmq_credentials = credentials.is_ibmq()
        kwargs.update({"qe_token": credentials.token, "qe_url": credentials.url})

        return func(self, *args, **kwargs)

    return _wrapper


def enforce_subclasses_call(
    methods: Union[str, Iterable[str]], attr: str = "_enforce_subclasses_call_cache"
) -> Callable[[Type], Type]:
    """Class decorator which enforces that if any subclasses define on of the ``methods``, they must
    call ``super().<method>()`` or face a ``ValueError`` at runtime.

    This is unlikely to be useful for concrete test classes, who are not normally subclassed.  It
    should not be used on user-facing code, because it prevents subclasses from being free to
    override parent-class behavior, even when the parent-class behavior is not needed.

    This adds behavior to the ``__init__`` and ``__init_subclass__`` methods of the class, in
    addition to the named methods of this class and all subclasses.  The checks could be averted in
    grandchildren if a child class overrides ``__init_subclass__`` without up-calling the decorated
    class's method, though this would typically break inheritance principles.

    Arguments:
        methods:
            Names of the methods to add the enforcement to.  These do not necessarily need to be
            defined in the class body, provided they are somewhere in the method-resolution tree.

        attr:
            The attribute which will be added to all instances of this class and subclasses, in
            order to manage the call enforcement.  This can be changed to avoid clashes.

    Returns:
        A decorator, which returns its input class with the class with the relevant methods modified
        to include checks, and injection code in the ``__init_subclass__`` method.
    """

    methods = {methods} if isinstance(methods, str) else set(methods)

    def initialize_call_memory(self, *_args, **_kwargs):
        """Add the extra attribute used for tracking the method calls."""
        setattr(self, attr, set())

    def save_call_status(name):
        """Decorator, whose return saves the fact that the top-level method call occurred."""

        def out(self, *_args, **_kwargs):
            getattr(self, attr).add(name)

        return out

    def clear_call_status(name):
        """Decorator, whose return clears the call status of the method ``name``.  This prepares the
        call tracking for the child class's method call."""

        def out(self, *_args, **_kwargs):
            getattr(self, attr).discard(name)

        return out

    def enforce_call_occurred(name):
        """Decorator, whose return checks that the top-level method call occurred, and raises
        ``ValueError`` if not.  Concretely, this is an assertion that ``save_call_status`` ran."""

        def out(self, *_args, **_kwargs):
            cache = getattr(self, attr)
            if name not in cache:
                classname = self.__name__ if isinstance(self, type) else type(self).__name__
                raise ValueError(
                    f"Parent '{name}' method was not called by '{classname}.{name}'."
                    f" Ensure you have put in calls to 'super().{name}()'."
                )

        return out

    def wrap_subclass_methods(cls):
        """Wrap all the ``methods`` of ``cls`` with the call-tracking assertions that the top-level
        versions of the methods were called (likely via ``super()``)."""
        # Only wrap methods who are directly defined in this class; if we're resolving to a method
        # higher up the food chain, then it will already have been wrapped.
        for name in set(cls.__dict__) & methods:
            wrap_method(
                cls,
                name,
                before=clear_call_status(name),
                after=enforce_call_occurred(name),
            )

    def decorator(cls):
        # Add a class-level memory on, so class methods will work as well.  Instances will override
        # this on instantiation, to keep the "namespace" of class- and instance-methods separate.
        initialize_call_memory(cls)
        # Do the extra bits after the main body of __init__ so we can check we're not overwriting
        # anything, and after __init_subclass__ in case the decorated class wants to influence the
        # creation of the subclass's methods before we get to them.
        wrap_method(cls, "__init__", after=initialize_call_memory)
        for name in methods:
            wrap_method(cls, name, before=save_call_status(name))
        wrap_method(cls, "__init_subclass__", after=wrap_subclass_methods)
        return cls

    return decorator


class _TestOptions(collections.abc.Mapping):
    """Lazy-loading view onto the test options retrieved from the environment."""

    __slots__ = ("_options",)

    def __init__(self):
        self._options = None

    def _load(self):
        if self._options is None:
            self._options = get_test_options()

    def __getitem__(self, key):
        self._load()
        return self._options[key]

    def __iter__(self):
        self._load()
        return iter(self._options)

    def __len__(self):
        self._load()
        return len(self._options)


TEST_OPTIONS = _TestOptions()
