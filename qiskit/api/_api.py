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


def register(token, url='https://quantumexperience.ng.bluemix.net/api',
             hub=None, group=None, project=None):
    """
    Register a user with an API. By calling this method, all available
    backends from this API are registered into QISKit.
    Defualt API is the IBM Q Experience.

    Args:
        token (str): user authentication token
        url (str): API's url
        hub (str): optional user hub
        group (str): optional user group
        project (str): optional user project

    Returns:
        API: an api object
    """
    config = {
        'url': url,
        'hub': hub,
        'group': group,
        'project': project
    }
    api_temp = IBMQuantumExperience(token, config)
    api = API(api_temp)
    qiskit.backends.discover_remote_backends(api_temp)

    # Ideally this would make an API object based on url and the user token
    # and register all the backends of this API to qiskit.backends.remote()
    # I am worried that there is not checks to see if the backends have the same name
    # this should be verified in the future.
    return api

class API(object):
    """Creates an API object."""

    # Functions to add
    #   status -- gives the status of the api
    #   available_backends -- all backends in this API, with their config
    # A use case is the user would do
    # ibmqx = qiskit.api.register(token,url)
    # ibmqx.status and it prints the current status of the API

    def __init__(self, api):
        """Create an API object."""

        # Ideally we should give this a url, but while we import the IBMQuantumExperience object
        # I think this the best until we bring functions from IBMQuantumExperience into this object
        self.api = api
