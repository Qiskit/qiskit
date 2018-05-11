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
"""Helper module for simplified QISKit usage."""

import qiskit._compiler
from qiskit import QISKitError
from qiskit.backends.ibmq.ibmqprovider import IBMQProvider
from qiskit.wrapper.defaultqiskitprovider import DefaultQISKitProvider
from qiskit import QuantumJob

# Default provider used by the rest of the functions on this module. Please
# note that this is a global object.
_DEFAULT_PROVIDER = DefaultQISKitProvider()


def register(token, url,
             hub=None, group=None, project=None, proxies=None, verify=True,
             provider_name='ibmq'):
    """
    Authenticate against an online backend provider.

    Args:
            token (str): The token used to register on the online backend such
                as the quantum experience.
            url (str): The url used for online backend such as the quantum
                experience.
            hub (str): The hub used for online backend.
            group (str): The group used for online backend.
            project (str): The project used for online backend.
            proxies (dict): Proxy configuration for the API, as a dict with
                'urls' and credential keys.
            verify (bool): If False, ignores SSL certificates errors.
            provider_name (str): the unique name for the online backend
                provider (for example, 'ibmq' for the IBM Quantum Experience).
    Raises:
        QISKitError: if the provider name is not recognized.
    """
    if provider_name == 'ibmq':
        provider = IBMQProvider(token, url,
                                hub, group, project, proxies, verify)
        _DEFAULT_PROVIDER.add_provider(provider)
    else:
        raise QISKitError('provider name %s is not recognized' % provider_name)


# Functions for inspecting and retrieving backends.


def available_backends(filters=None, compact=True):
    """
    Return the backends that are available in the SDK, optionally filtering
    them based on their capabilities.

    Note:
        In order for this function to return online backends, a connection with
        an online backend provider needs to be established by calling the
        `register()` function.
    Args:
        filters (dict): dictionary of filtering conditions.
        compact (bool): group backend names based on aliases

    Returns:
        list[str]: the names of the available backends.
    """
    backend_names = [str(backend)
                     for backend in _DEFAULT_PROVIDER.available_backends(filters)]

    if compact:
        alias_dict = _DEFAULT_PROVIDER.aliased_backend_names()
        aliases = set()
        for name in backend_names:
            backend_alias = set(k for k, v in alias_dict.items() if name in v)
            if not backend_alias:
                aliases.add(name)
            elif len(backend_alias) == 1:
                (alias,) = backend_alias
                aliases.add(alias)
        backend_names = list(aliases)

    return backend_names


def local_backends(compact=True):
    """
    Return the available local backends.

    Args:
        compact (bool): only report alias names. this is usually shorter, any several
        backends usually share the same alias.

    Returns:
        list[str]: the names of the available remote backends.
    """
    return available_backends({'local': True}, compact=compact)


def remote_backends(compact=True):
    """
    Return the available remote backends.

    Args:
        compact (bool): only report alias names. this is usually shorter, any several
        backends usually share the same alias.

    Returns:
        list[str]: the names of the available remote backends.
    """
    return available_backends({'local': False}, compact=compact)


def get_backend(name):
    """
    Return an instance of a `Backend` object from its name identifier.

    Args:
        name(str): unique name of the backend.
    Returns:
        BaseBackend: a Backend instance.
    """
    return _DEFAULT_PROVIDER.get_backend(name)


# Functions for compiling and executing.


def compile(circuits, backend,
            config=None, basis_gates=None, coupling_map=None, initial_layout=None,
            shots=1024, max_credits=10, seed=None, qobj_id=None, hpc=None,
            skip_translation=False):
    """Compile a list of circuits into a qobj.

    Args:
        circuits (QuantumCircuit or list[QuantumCircuit]): circuits to compile
        backend (BaseBackend or str): a backend to compile for
        config (dict): dictionary of parameters (e.g. noise) used by runner
        basis_gates (str): comma-separated basis gate set to compile to
        coupling_map (list): coupling map (perhaps custom) to target in mapping
        initial_layout (list): initial layout of qubits in mapping
        shots (int): number of repetitions of each circuit, for sampling
        max_credits (int): maximum credits to use
        seed (int): random seed for simulators
        qobj_id (int): identifier for the generated qobj
        hpc (dict): HPC simulator parameters
        skip_translation (bool): If True, bypass most of the compilation process and
            creates a qobj with minimal check nor translation
    Returns:
        obj: the qobj to be run on the backends
    """
    # pylint: disable=redefined-builtin
    if isinstance(backend, str):
        backend = _DEFAULT_PROVIDER.get_backend(backend)
    return qiskit._compiler.compile(circuits, backend,
                                    config, basis_gates, coupling_map, initial_layout,
                                    shots, max_credits, seed, qobj_id, hpc,
                                    skip_translation)


def execute(circuits, backend,
            config=None, basis_gates=None, coupling_map=None, initial_layout=None,
            shots=1024, max_credits=10, seed=None, qobj_id=None, hpc=None,
            skip_translation=False):
    """Executes a set of circuits.

    Args:
        circuits (QuantumCircuit or list[QuantumCircuit]): circuits to execute
        backend (BaseBackend or str): a backend to execute the circuits on
        config (dict): dictionary of parameters (e.g. noise) used by runner
        basis_gates (str): comma-separated basis gate set to compile to
        coupling_map (list): coupling map (perhaps custom) to target in mapping
        initial_layout (list): initial layout of qubits in mapping
        shots (int): number of repetitions of each circuit, for sampling
        max_credits (int): maximum credits to use
        seed (int): random seed for simulators
        qobj_id (int): identifier for the generated qobj
        hpc (dict): HPC simulator parameters
        skip_translation (bool): skip most of the compile steps and produce qobj directly

    Returns:
        BaseJob: returns job instance derived from BaseJob
    """
    if isinstance(backend, str):
        backend = _DEFAULT_PROVIDER.get_backend(backend)
    qobj = compile(circuits, backend,
                   config, basis_gates, coupling_map, initial_layout,
                   shots, max_credits, seed, qobj_id, hpc,
                   skip_translation)
    # XXX When qobj is done this should replace q_job
    q_job = QuantumJob(qobj, backend=backend, preformatted=True, resources={
        'max_credits': qobj['config']['max_credits']})
    return backend.run(q_job)
