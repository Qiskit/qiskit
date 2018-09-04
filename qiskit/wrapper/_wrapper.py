# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Helper module for simplified QISKit usage."""

import logging
import warnings
from qiskit import transpiler, QISKitError
from qiskit.backends.ibmq import IBMQProvider
from qiskit.wrapper import credentials
from qiskit.wrapper.defaultqiskitprovider import DefaultQISKitProvider
from qiskit._util import _parse_ibmq_credentials
from ._circuittoolkit import circuit_from_qasm_file, circuit_from_qasm_string

# Default provider used by the rest of the functions on this module. Please
# note that this is a global object.
_DEFAULT_PROVIDER = DefaultQISKitProvider()

logger = logging.getLogger(__name__)


def register(*args, provider_class=IBMQProvider, **kwargs):
    """
    Authenticate against an online backend provider.
    This is a factory method that returns the provider that gets registered.

    Note that if no parameters are passed, this method will try to
    automatically discover the credentials for IBMQ in the following places,
    in order::

        1. in the `Qconfig.py` file in the current working directory.
        2. in the environment variables.
        3. in the `qiskitrc` configuration file.

    Args:
        args (tuple): positional arguments passed to provider class initialization
        provider_class (BaseProvider): provider class
        kwargs (dict): keyword arguments passed to provider class initialization.
            For the IBMQProvider default this can include things such as:

                * token (str): The token used to register on the online backend such
                    as the quantum experience.
                * url (str): The url used for online backend such as the quantum
                    experience.
                * hub (str): The hub used for online backend.
                * group (str): The group used for online backend.
                * project (str): The project used for online backend.
                * proxies (dict): Proxy configuration for the API, as a dict with
                    'urls' and credential keys.
                * verify (bool): If False, ignores SSL certificates errors.

    Returns:
        BaseProvider: the provider instance that was just registered.

    Raises:
        QISKitError: if the provider could not be registered (e.g. due to
        conflict, or if no credentials were provided.)
    """
    # Try to autodiscover credentials if not passed.
    if not args and not kwargs and provider_class == IBMQProvider:
        kwargs = credentials.discover_credentials().get(
            credentials.get_account_name(IBMQProvider)) or {}
        if not kwargs:
            raise QISKitError(
                'No IBMQ credentials found. Please pass them explicitly or '
                'store them before calling register() with store_credentials()')

    try:
        provider = provider_class(*args, **kwargs)
    except Exception as ex:
        raise QISKitError("Couldn't instantiate provider! Error: {0}".format(ex))

    _DEFAULT_PROVIDER.add_provider(provider)
    return provider


def unregister(provider):
    """
    Removes a provider from list of registered providers.

    Note:
        If backend names from provider1 and provider2 were clashing,
        `unregister(provider1)` removes the clash and makes the backends
        from provider2 available.

    Args:
        provider (BaseProvider): the provider instance to unregister
    Raises:
        QISKitError: if the provider instance is not registered
    """
    _DEFAULT_PROVIDER.remove_provider(provider)


def registered_providers():
    """Return the currently registered providers."""
    return list(_DEFAULT_PROVIDER.providers)


def store_credentials(token, url='https://quantumexperience.ng.bluemix.net/api',
                      hub=None, group=None, project=None, proxies=None,
                      verify=True, overwrite=False):
    """
    Store credentials for the IBMQ account in the config file.

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
        overwrite (bool): overwrite existing credentials.

    Raises:
        QISKitError: if the credentials already exist and overwrite==False.
    """
    url = _parse_ibmq_credentials(url, hub, group, project)
    credentials.store_credentials(
        provider_class=IBMQProvider, overwrite=overwrite,
        token=token, url=url, proxies=proxies, verify=verify)


# Functions for inspecting and retrieving backends.
def available_backends(filters=None, compact=True):
    """
    Return names of backends that are available in the SDK, optionally filtering
    them based on their capabilities.

    Note:
        In order for this function to return online backends, a connection with
        an online backend provider needs to be established by calling the
        `register()` function.

    Note:
        If two or more providers have backends with the same name, those names
        will be shown only once. To disambiguate and choose a backend from a
        specific provider, get the backend from that specific provider.

        Example::

            p1 = register(token1)
            p2 = register(token2)
            execute(circuit, p1.get_backend('ibmq_5_tenerife'))
            execute(circuit, p2.get_backend('ibmq_5_tenerife'))

    Args:
        filters (dict or callable): filtering conditions.
        compact (bool): group backend names based on compact group names.

    Returns:
        list[str]: the names of the available backends.
    """
    backend_names = [str(backend)
                     for backend in _DEFAULT_PROVIDER.available_backends(filters)]

    alias_dict = {v: k for k, v in _DEFAULT_PROVIDER.aliased_backend_names().items()}
    backend_names = [alias_dict[name] if name in alias_dict else name for name in backend_names]

    if compact:
        group_dict = _DEFAULT_PROVIDER.grouped_backend_names()
        groups = set()
        for name in backend_names:
            backend_group = set(k for k, v in group_dict.items() if name in v)
            if not backend_group:
                groups.add(name)
            elif len(backend_group) == 1:
                (group,) = backend_group
                groups.add(group)
        backend_names = list(groups)

    return sorted(backend_names)


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
        return min([b for b in backends
                    if b.status()['operational'] and 'pending_jobs' in b.status()],
                   key=lambda b: b.status()['pending_jobs']).name()
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
        Qobj: the qobj to be run on the backends
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
