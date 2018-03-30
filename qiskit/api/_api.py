# -*- coding: utf-8 -*-
# pylint: disable=missing-param-doc,missing-type-doc
#
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

"""API class"""
import qiskit.backends
from IBMQuantumExperience import IBMQuantumExperience


def register(token, url='https://quantumexperience.ng.bluemix.net/api'):
    """Return the filename.
    """
    api_temp = IBMQuantumExperience(token, config={'url': url})
    api = API(api_temp)
    qiskit.backends.discover_remote_backends(api_temp)

    # Ideally this would make a API object based on url and the user token
    # and register all the backends of this API to qiskit.backends.remote()
    # I am worried that there is not checks is the backends have the same name
    # this should be verified in the future.
    return api

class API(object):
    """Creates a API object."""

    # Functions to add status -- gives the status of the api
    # A use case is the user would do
    # ibmqx = qiskit.api.register(token,url)
    # ibmqx.status and it prints the current status of the API

    def __init__(self, api):
        """Create an API object."""

        # Ideally we should give this a url, but while we import the IBMQuantumExperience object
        # i think this the best until we bring functions for IBMQuantumExperience into this object
        self.api = api
