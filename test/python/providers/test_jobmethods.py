# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=missing-docstring

"""BasicAerJob creation and test suite."""

import uuid
from contextlib import contextmanager
from os import path
import unittest

from unittest.mock import patch
from qiskit.test import QiskitTestCase
from qiskit.test.mock import new_fake_qobj, FakeRueschlikon


class TestSimulatorsJob(QiskitTestCase):
    """Test how backends create BasicAerJob objects and the BasicAerJob class."""

    def test_multiple_execution(self):
        # Notice that it is Python responsibility to test the executors
        # can run several tasks at the same time. It is our responsibility to
        # use the executor correctly. That is what this test checks.

        taskcount = 10
        target_tasks = [lambda: None for _ in range(taskcount)]

        job_id = str(uuid.uuid4())
        backend = FakeRueschlikon()
        # pylint: disable=invalid-name,redefined-outer-name
        with mocked_executor() as (SimulatorJob, executor):
            for index in range(taskcount):
                job = SimulatorJob(backend, job_id, target_tasks[index], new_fake_qobj())
                job.submit()

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

        job_id = str(uuid.uuid4())
        backend = FakeRueschlikon()
        # pylint: disable=invalid-name,redefined-outer-name
        with mocked_executor() as (BasicAerJob, executor):
            job = BasicAerJob(backend, job_id, lambda: None, new_fake_qobj())
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


@contextmanager
def mocked_executor():
    """Context that patches the derived executor classes to return the same
    executor object. Also patches the future object returned by executor's
    submit()."""

    import importlib
    import concurrent.futures as futures
    import qiskit.providers.basicaer.basicaerjob as basicaerjob

    executor = unittest.mock.MagicMock(spec=futures.Executor)
    executor.submit.return_value = unittest.mock.MagicMock(spec=futures.Future)
    mock_options = {'return_value': executor, 'autospec': True}
    with patch.object(futures, 'ProcessPoolExecutor', **mock_options),\
            patch.object(futures, 'ThreadPoolExecutor', **mock_options):
        importlib.reload(basicaerjob)
        yield basicaerjob.BasicAerJob, executor


@contextmanager
def mocked_simulator_binaries():
    """Context to force binary-based simulators to think the simulators exist.
    """
    with patch.object(path, 'exists', return_value=True, autospec=True),\
            patch.object(path, 'getsize', return_value=1000, autospec=True):
        yield


if __name__ == '__main__':
    unittest.main(verbosity=2)
