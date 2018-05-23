# -*- coding: utf-8 -*-
# pylint: disable=invalid-name,missing-docstring,broad-except

# Copyright 2017 IBM RESEARCH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""LocalJob creation and test suite."""

from os import path
import unittest
from unittest.mock import patch
from contextlib import contextmanager
from qiskit.backends.local import LocalJob
from qiskit.backends.local import QasmSimulatorCpp, QasmSimulatorProjectQ
from qiskit.backends.local import QasmSimulatorPy
from qiskit.backends.local import StatevectorSimulatorCpp
from qiskit.backends.local import StatevectorSimulatorPy
from qiskit.backends.local import StatevectorSimulatorSympy
from qiskit.backends.local import UnitarySimulatorPy, UnitarySimulatorSympy
from .common import QiskitTestCase


class TestLocalJob(QiskitTestCase):
    """Test how backends create LocalJob objects and the LocalJob class."""

    _backends = [
        QasmSimulatorCpp,
        QasmSimulatorProjectQ,
        QasmSimulatorPy,
        StatevectorSimulatorCpp,
        StatevectorSimulatorPy,
        StatevectorSimulatorSympy,
        UnitarySimulatorPy,
        UnitarySimulatorSympy
    ]

    def test_run(self):
        with mocked_simulator_binaries(),\
             patch.object(LocalJob, '__init__', return_value=None,
                          autospec=True):

            for backendConstructor in self._backends:
                self.log.info('Backend under test: %s', backendConstructor)
                backend = backendConstructor()
                job = backend.run(FakeQJob())
                self.assertIsInstance(job, LocalJob)

    def test_multiple_execution(self):
        # Notice that it is Python responsibility to test the executors
        # can run several tasks at the same time. It is our responsibility to
        # use the executor correctly. That is what this test checks.

        taskcount = 10
        target_tasks = [lambda: None for _ in range(taskcount)]

        # pylint: disable=redefined-outer-name
        with intercepted_executor_for_LocalJob() as (LocalJob, executor):
            for index in range(taskcount):
                LocalJob(target_tasks[index], FakeQJob())

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

        # pylint: disable=redefined-outer-name
        with intercepted_executor_for_LocalJob() as (LocalJob, executor):
            job = LocalJob(lambda: None, FakeQJob())
            job.cancel()

        executor.submit.assert_called_once()
        mockedFuture = executor.submit.return_value
        mockedFuture.cancel.assert_called_once()

    def test_done(self):
        # Like before.
        # pylint: disable=redefined-outer-name
        with intercepted_executor_for_LocalJob() as (LocalJob, executor):
            job = LocalJob(lambda: None, FakeQJob())
            _ = job.done

        executor.submit.assert_called_once()
        mockedFuture = executor.submit.return_value
        mockedFuture.done.assert_called_once()


class FakeQJob():
    def __init__(self):
        self.backend = FakeBackend()
        self.qobj = {
            'id': 'test-id',
            'config': {
                'backend_name': self.backend.name,
                'shots': 1024,
                'max_credits': 100
            },
            'circuits': [{
                'compiled_circuit_qasm': 'fake-code',
                'config': {
                    'seed': 123456
                }
            }]
        }


class FakeBackend():

    def __init__(self):
        self.name = 'test-backend'


@contextmanager
def intercepted_executor_for_LocalJob():
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
    from qiskit.backends.local import qasm_simulator_projectq
    with patch.object(path, 'exists', return_value=True, autospec=True),\
            patch.object(path, 'getsize', return_value=1000, autospec=True),\
            patch.object(qasm_simulator_projectq, 'CppSim', {}):
        yield


if __name__ == '__main__':
    unittest.main(verbosity=2)
