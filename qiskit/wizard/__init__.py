# -*- coding: utf-8 -*-

# Copyright 2018 IBM RESEARCH. All Rights Reserved.
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
"""Helper for simplified QISKit usage."""

import qiskit._compiler
from qiskit import QISKitError
from qiskit.backends.ibmq.ibmqprovider import IBMQProvider
from qiskit.wizard.defaultqiskitprovider import DefaultQISKitProvider

# THIS IS A GLOBAL OBJECT - USE WITH CARE.
DEFAULT_PROVIDER = DefaultQISKitProvider()


def register(token, url,
             hub=None, group=None, project=None, proxies=None, verify=True,
             provider_name='qiskit'):
    """
    Authenticate against an online provider.

    Raises:
        QISKitError: if the provider is not recognized.
    """
    if provider_name == 'qiskit':
        provider = IBMQProvider(token, url,
                                hub, group, project, proxies, verify)
        DEFAULT_PROVIDER.add_provider(provider)
    else:
        raise QISKitError('provider name %s is not recognized' % provider_name)

# Functions for inspecting and retrieving backends.


def available_backends(filters=None):
    """
    Return available backends.

    Args:
        filters (dict): dictionary of filtering conditions.
    Returns:
        list: (of str): the names of the available backends.
    """
    return DEFAULT_PROVIDER.available_backends(filters)


def local_backends():
    """Get the local backends."""
    return available_backends({'local': True})


def remote_backends():
    """Get the remote backends."""
    return available_backends({'local': False})


def get_backend(name):
    """
    Return a backend.
    """
    return DEFAULT_PROVIDER.get_backend(name)


# Functions for compiling and executing.


def compile(list_of_circuits, backend_id, compile_config=None):
    """Compile a list of circuits into a qobj.

    Args:
        list_of_circuits (list[QuantumCircuits]): list of circuits
        backend_id (str): a backend object to use as the default compiling option
        compile_config (dict or None): a dictionary of compile configurations.
            If `None`, the default compile configuration will be used.
    Returns:
        obj: the qobj to be run on the backends
    """
    # pylint: disable=redefined-builtin
    backend = DEFAULT_PROVIDER.get_backend(backend_id)
    return qiskit._compiler.compile(list_of_circuits, backend, compile_config)


def execute(list_of_circuits, backend_id, compile_config=None,
            wait=5, timeout=60):
    """Executes a set of circuits.

    Args:
        list_of_circuits (list[QuantumCircuits]): list of circuits
        backend_id (str): A string for the backend name to use
        compile_config (dict or None): a dictionary of compile configurations.
        wait (int): XXX -- I DONT THINK WE NEED TO KEEP THIS
        timeout (int): XXX -- I DONT THINK WE NEED TO KEEP THIS

    Returns:
        obj: The results object
    """
    backend = DEFAULT_PROVIDER.get_backend(backend_id)
    return qiskit._compiler.execute(list_of_circuits, backend, compile_config,
                                    wait, timeout)
