# -*- coding: utf-8 -*-
# pylint: disable=invalid-name,missing-docstring

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
"""Test API hub, groups and projects related functionality."""
import os
from unittest import skipIf
from unittest.mock import patch

from qiskit import QuantumProgram
from IBMQuantumExperience.IBMQuantumExperience import _Request

from .common import QiskitTestCase


# We need the environment variable for Travis.
HAS_GROUP_VARS = False
try:
    import Qconfig

    QE_TOKEN = Qconfig.APItoken
    QE_URL = Qconfig.config['url']
    QE_HUB = Qconfig.config['hub']
    QE_GROUP = Qconfig.config['group']
    QE_PROJECT = Qconfig.config['project']
    HAS_GROUP_VARS = True
except ImportError:
    if all(var in os.environ for var in
           ['QE_TOKEN', 'QE_URL', 'QE_HUB', 'QE_GROUP', 'QE_PROJECT']):
        QE_TOKEN = os.environ['QE_TOKEN']
        QE_URL = os.environ['QE_URL']
        QE_HUB = os.environ['QE_HUB']
        QE_GROUP = os.environ['QE_GROUP']
        QE_PROJECT = os.environ['QE_PROJECT']
        HAS_GROUP_VARS = True


class TestApiHub(QiskitTestCase):
    """Tests for API hubs, groups, and projects."""

    def setUp(self):
        super(TestApiHub, self).setUp()
        self.backend = 'ibmqx4'

    def _get_quantum_program(self):
        quantum_program = QuantumProgram()
        qr = quantum_program.create_quantum_register("q", 1)
        cr = quantum_program.create_classical_register("c", 1)
        qc = quantum_program.create_circuit("qc", [qr], [cr])
        qc.h(qr[0])
        qc.measure(qr[0], cr[0])

        return quantum_program

    @skipIf(not HAS_GROUP_VARS, 'QE group variables not present')
    def test_execute_api_no_parameters(self):
        """Test calling the API with no hub parameters."""
        quantum_program = self._get_quantum_program()

        # Invoke with no hub, group or project parameters.
        quantum_program.set_api(QE_TOKEN, QE_URL)

        # Store the original post() method.
        post_original = quantum_program._QuantumProgram__api.req.post
        with patch.object(_Request, 'post',
                          wraps=post_original) as mocked_post:
            _ = quantum_program.execute(
                ['qc'], backend=self.backend, shots=1, max_credits=3)

            # Get the first parameter of the `run_job` POST call.
            url = mocked_post.call_args_list[-1][0][0]
            self.assertEqual('/Jobs', url)

    @skipIf(not HAS_GROUP_VARS, 'QE group variables not present')
    def test_execute_api_parameters(self):
        """Test calling the API with hub parameters."""
        quantum_program = self._get_quantum_program()

        # Invoke with hub, group and project parameters.
        quantum_program.set_api(QE_TOKEN, QE_URL, QE_HUB, QE_GROUP, QE_PROJECT)

        # Store the original post() method.
        post_original = quantum_program._QuantumProgram__api.req.post
        with patch.object(_Request, 'post',
                          wraps=post_original) as mocked_post:
            _ = quantum_program.execute(
                ['qc'], backend=self.backend, shots=1, max_credits=3)

            # Get the first parameter of the `run_job` POST call.
            url = mocked_post.call_args_list[-1][0][0]
            self.assertEqual('/Network/%s/Groups/%s/Projects/%s/jobs' %
                             (QE_HUB, QE_GROUP, QE_PROJECT),
                             url)

    @skipIf(not HAS_GROUP_VARS, 'QE group variables not present')
    def test_execute_api_modified_parameters(self):
        """Test calling the API with modified hub parameters."""
        quantum_program = self._get_quantum_program()

        # Invoke with hub, group and project parameters.
        FAKE_QE_HUB = 'HUB'
        FAKE_QE_GROUP = 'GROUP'
        FAKE_QE_PROJECT = 'PROJECT'
        quantum_program.set_api(QE_TOKEN, QE_URL,
                                FAKE_QE_HUB, FAKE_QE_GROUP, FAKE_QE_PROJECT)

        # Modify the api parameters.
        self.assertEqual(quantum_program._QuantumProgram__api.config['hub'],
                         FAKE_QE_HUB)
        quantum_program.set_api_hubs_config(QE_HUB, QE_GROUP, QE_PROJECT)
        self.assertEqual(quantum_program._QuantumProgram__api.config['hub'],
                         QE_HUB)

        # Store the original post() method.
        post_original = quantum_program._QuantumProgram__api.req.post
        with patch.object(_Request, 'post',
                          wraps=post_original) as mocked_post:
            _ = quantum_program.execute(
                ['qc'], backend=self.backend, shots=1, max_credits=3)

            # Get the first parameter of the `run_job` POST call.
            url = mocked_post.call_args_list[-1][0][0]
            self.assertEqual('/Network/%s/Groups/%s/Projects/%s/jobs' %
                             (QE_HUB, QE_GROUP, QE_PROJECT),
                             url)

    @skipIf(not HAS_GROUP_VARS, 'QE group variables not present')
    def test_api_calls_no_parameters(self):
        """Test calling some endpoints of the API with no hub parameters.

        Note: this tests does not make assertions, it is only intended to
            verify the endpoints.
        """
        quantum_program = self._get_quantum_program()

        # Invoke with no hub, group or project parameters.
        quantum_program.set_api(QE_TOKEN, QE_URL)

        self.log.info(quantum_program.online_backends())
        self.log.info(quantum_program.get_backend_parameters(self.backend))
        self.log.info(quantum_program.get_backend_calibration(self.backend))

    @skipIf(not HAS_GROUP_VARS, 'QE group variables not present')
    def test_api_calls_parameters(self):
        """Test calling some endpoints of the API with hub parameters.

        Note: this tests does not make assertions, it is only intended to
            verify the endpoints.
        """
        quantum_program = self._get_quantum_program()

        # Invoke with hub, group and project parameters.
        quantum_program.set_api(QE_TOKEN, QE_URL, QE_HUB, QE_GROUP, QE_PROJECT)

        self.log.info(quantum_program.online_backends())
        self.log.info(quantum_program.get_backend_parameters(self.backend))
        self.log.info(quantum_program.get_backend_calibration(self.backend))
