# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Base class for dummy backends.
"""

import uuid
import time

from qiskit.providers.models import BackendProperties
from qiskit.providers import BaseBackend
from qiskit.result import Result
from .fake_job import FakeJob


class FakeBackend(BaseBackend):
    """This is a dummy backend just for testing purposes."""

    def __init__(self, configuration, time_alive=10):
        """FakeBackend initializer.

        Args:
            configuration (BackendConfiguration): backend configuration
            time_alive (int): time to wait before returning result
        """
        super().__init__(configuration)
        self.time_alive = time_alive

    def properties(self):
        """Return backend properties"""
        coupling_map = self.configuration().coupling_map
        unique_qubits = list(set().union(*coupling_map))

        properties = {
            'backend_name': self.name(),
            'backend_version': self.configuration().backend_version,
            'last_update_date': '2000-01-01 00:00:00Z',
            'qubits': [
                [
                    {
                        "date": "2000-01-01 00:00:00Z",
                        "name": "T1",
                        "unit": "\u00b5s",
                        "value": 0.0
                    },
                    {
                        "date": "2000-01-01 00:00:00Z",
                        "name": "T2",
                        "unit": "\u00b5s",
                        "value": 0.0
                    },
                    {
                        "date": "2000-01-01 00:00:00Z",
                        "name": "frequency",
                        "unit": "GHz",
                        "value": 0.0
                    },
                    {
                        "date": "2000-01-01 00:00:00Z",
                        "name": "readout_error",
                        "unit": "",
                        "value": 0.0
                    }
                ] for _ in range(len(unique_qubits))
            ],
            'gates': [{
                "gate": "cx",
                "name": "CX" + str(pair[0]) + "_" + str(pair[1]),
                "parameters": [
                    {
                        "date": "2000-01-01 00:00:00Z",
                        "name": "gate_error",
                        "unit": "",
                        "value": 0.0
                    }
                ],
                "qubits": [
                    pair[0],
                    pair[1]
                ]
            } for pair in coupling_map],
            'general': []
        }

        return BackendProperties.from_dict(properties)

    def run(self, qobj):
        job_id = str(uuid.uuid4())
        job = FakeJob(self, job_id, self.run_job, qobj)
        job.submit()
        return job

    def run_job(self, job_id, qobj):
        """Main dummy run loop"""
        del qobj  # unused
        time.sleep(self.time_alive)

        return Result.from_dict(
            {'job_id': job_id, 'result': [], 'status': 'COMPLETED'})
