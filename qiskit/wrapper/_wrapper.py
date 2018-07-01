# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Helper module for simplified QISKit usage."""

import warnings
from qiskit import transpiler, QISKitError

from qiskit.wrapper.defaultqiskitprovider import DefaultQISKitProvider
from ._circuittoolkit import circuit_from_qasm_file, circuit_from_qasm_string


# Default provider used by the rest of the functions on this module. Please
# note that this is a global object.
_DEFAULT_PROVIDER = DefaultQISKitProvider()


def register(token, url='https://quantumexperience.ng.bluemix.net/api',
             hub=None, group=None, project=None, proxies=None, verify=True,
             provider_name=None):
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
            provider_name (str): the user-provided name for the registered
                provider.
    Raises:
        QISKitError: if the provider name is not recognized.
    """
    # Convert the credentials to a dict.
    credentials = {
        'token': token, 'url': url, 'hub': hub, 'group': group,
        'project': project, 'proxies': proxies, 'verify': verify
    }
    _DEFAULT_PROVIDER.add_ibmq_provider(credentials, provider_name)


def unregister(provider_name):
    """
    Removes a provider of list of registered providers.

    Args:
        provider_name (str): The unique name for the online provider.
    Raises:
        QISKitError: if the provider name is not valid.
    """
    _DEFAULT_PROVIDER.remove_provider(provider_name)


def registered_providers():
    """Return the names of the currently registered providers."""
    return list(_DEFAULT_PROVIDER.providers.keys())


# Functions for inspecting and retrieving backends.


def available_backends(filters=None, compact=True):
    """
    Return names of backends that are available in the SDK, optionally filtering
    them based on their capabilities.

    Note:
        In order for this function to return online backends, a connection with
        an online backend provider needs to be established by calling the
        `register()` function.
    Args:
        filters (dict or callable): filtering conditions.
        compact (bool): group backend names based on compact group names.

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
    warnings.warn(
        "local_backends() will be deprecated in upcoming versions (>0.5). "
        "using filters instead is recommended (i.e. available_backends({'local': True}).",
        DeprecationWarning)
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
    warnings.warn(
        "remote_backends() will be deprecated in upcoming versions (>0.5). "
        "using filters instead is recommended (i.e. available_backends({'local': False}).",
        DeprecationWarning)
    return available_backends({'local': False}, compact=compact)


def least_busy(names):
    """
    Return the least busy available backend for those that
    have a `pending_jobs` in their `status`. Backends such as
    local backends that do not have this are not considered.

    Args:
        names (list[str]): backend names to choose from
                    (e.g. output of ``available_backends()``)

    Returns:
        str: the name of the least busy backend

    Raises:
        QISKitError: if passing a list of backend names that is
            either empty or none have attribute ``pending_jobs``
    """
    backends = [get_backend(name) for name in names]
    try:
        return min([b for b in backends if b.status['operational'] and 'pending_jobs' in b.status],
                   key=lambda b: b.status['pending_jobs']).name
    except (ValueError, TypeError):
        raise QISKitError("Can only find least_busy backend from a non-empty list.")


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
            skip_transpiler=False):
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
        skip_transpiler (bool): If True, bypass most of the compilation process and
            creates a qobj with minimal check nor translation
    Returns:
        obj: the qobj to be run on the backends
    """
    # pylint: disable=redefined-builtin
    if isinstance(backend, str):
        backend = _DEFAULT_PROVIDER.get_backend(backend)

    pass_manager = None  # default pass manager which executes predetermined passes
    if skip_transpiler:  # empty pass manager which does nothing
        pass_manager = transpiler.PassManager()

    return transpiler.compile(circuits, backend,
                              config, basis_gates, coupling_map, initial_layout,
                              shots, max_credits, seed, qobj_id, hpc,
                              pass_manager)


def execute(circuits, backend,
            config=None, basis_gates=None, coupling_map=None, initial_layout=None,
            shots=1024, max_credits=10, seed=None, qobj_id=None, hpc=None,
            skip_transpiler=False):
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
        skip_transpiler (bool): skip most of the compile steps and produce qobj directly

    Returns:
        BaseJob: returns job instance derived from BaseJob
    """
    # pylint: disable=missing-param-doc, missing-type-doc
    if isinstance(backend, str):
        backend = _DEFAULT_PROVIDER.get_backend(backend)
    qobj = compile(circuits, backend,
                   config, basis_gates, coupling_map, initial_layout,
                   shots, max_credits, seed, qobj_id, hpc,
                   skip_transpiler)
    return backend.run(qobj)


# Functions for importing qasm


def load_qasm_string(qasm_string, name=None,
                     basis_gates="id,u0,u1,u2,u3,x,y,z,h,s,sdg,t,tdg,rx,ry,rz,"
                                 "cx,cy,cz,ch,crz,cu1,cu3,swap,ccx,cswap"):
    """Construct a quantum circuit from a qasm representation (string).

    Args:
        qasm_string (str): a string of qasm, or a filename containing qasm.
        basis_gates (str): basis gates for the quantum circuit.
        name (str or None): the name of the quantum circuit after loading qasm
            text into it. If no name given, assign automatically.
    Returns:
        QuantumCircuit: circuit constructed from qasm.
    Raises:
        QISKitError: if the string is not valid QASM
    """
    return circuit_from_qasm_string(qasm_string, name, basis_gates)


def load_qasm_file(qasm_file, name=None,
                   basis_gates="id,u0,u1,u2,u3,x,y,z,h,s,sdg,t,tdg,rx,ry,rz,"
                               "cx,cy,cz,ch,crz,cu1,cu3,swap,ccx,cswap"):
    """Construct a quantum circuit from a qasm representation (file).

    Args:
        qasm_file (str): a string for the filename including its location.
        name (str or None): the name of the quantum circuit after
            loading qasm text into it. If no name is give the name is of
            the text file.
        basis_gates (str): basis gates for the quantum circuit.
    Returns:
         QuantumCircuit: circuit constructed from qasm.
    Raises:
        QISKitError: if the file cannot be read.
    """
    return circuit_from_qasm_file(qasm_file, name, basis_gates)
