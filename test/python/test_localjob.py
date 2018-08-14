# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=missing-docstring

"""LocalJob creation and test suite."""

from contextlib import contextmanager
from os import path
import unittest
from unittest.mock import patch

from qiskit.backends.local import QasmSimulatorCpp
from qiskit.backends.local import QasmSimulatorPy
from qiskit.backends.local import StatevectorSimulatorCpp
from qiskit.backends.local import StatevectorSimulatorPy
from qiskit.backends.local import UnitarySimulatorPy
from .common import QiskitTestCase
from ._mockutils import new_fake_qobj


class TestLocalJob(QiskitTestCase):
    """Test how backends create LocalJob objects and the LocalJob class."""

    _backends = [
        QasmSimulatorCpp,
        QasmSimulatorPy,
        StatevectorSimulatorCpp,
        StatevectorSimulatorPy,
        UnitarySimulatorPy
    ]

    def test_multiple_execution(self):
        # Notice that it is Python responsibility to test the executors
        # can run several tasks at the same time. It is our responsibility to
        # use the executor correctly. That is what this test checks.

        taskcount = 10
        target_tasks = [lambda: None for _ in range(taskcount)]

        # pylint: disable=invalid-name,redefined-outer-name
        with mocked_executor() as (LocalJob, executor):
            for index in range(taskcount):
                local_job = LocalJob(target_tasks[index], new_fake_qobj())
                local_job.submit()

        self.assertEqual(executor.submit.call_count, taskcount)
        for index in range(taskcount):
            _, callargs, _ = executor.submit.mock_calls[index]
            submitted_task = callargs[0]
            target_task = target_tasks[index]
            self.assertEqual(submitted_task, target_task)

    def test_cancel(self):
        # Again, cancelling jobs is beyond our responsibility. In this test
        # we only check if we delegate on the proper method of the underlaying
        # future object.

        # pylint: disable=invalid-name,redefined-outer-name
        with mocked_executor() as (LocalJob, executor):
            job = LocalJob(lambda: None, new_fake_qobj())
            job.submit()
            job.cancel()

        self.assertCalledOnce(executor.submit)
        mocked_future = executor.submit.return_value
        self.assertCalledOnce(mocked_future.cancel)

    def assertCalledOnce(self, mocked_callable):
        """Assert a mocked callable has been called once."""
        call_count = mocked_callable.call_count
        self.assertEqual(
            call_count, 1,
            'Callable object has been called more than once ({})'.format(
                call_count))


class FakeBackend():

    def __init__(self):
        self.name = 'test-backend'


@contextmanager
def mocked_executor():
    """Context that patches the derived executor classes to return the same
    executor object. Also patches the future object returned by executor's
    submit()."""

    import importlib
    import concurrent.futures as futures
    import qiskit.backends.local.localjob as localjob

    executor = unittest.mock.MagicMock(spec=futures.Executor)
    executor.submit.return_value = unittest.mock.MagicMock(spec=futures.Future)
    mock_options = {'return_value': executor, 'autospec': True}
    with patch.object(futures, 'ProcessPoolExecutor', **mock_options),\
            patch.object(futures, 'ThreadPoolExecutor', **mock_options):
        importlib.reload(localjob)
        yield localjob.LocalJob, executor


@contextmanager
def mocked_simulator_binaries():
    """Context to force binary-based simulators to think the simulators exist.
    """
    with patch.object(path, 'exists', return_value=True, autospec=True),\
            patch.object(path, 'getsize', return_value=1000, autospec=True):
        yield


if __name__ == '__main__':
    unittest.main(verbosity=2)
