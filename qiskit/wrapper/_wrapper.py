# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Helper module for simplified QISKit usage."""

import logging
import warnings
from qiskit import transpiler, QISKitError
from qiskit.backends.baseprovider import BaseProvider
from qiskit._util import _parse_ibmq_credentials
import qiskit.wrapper._register as reg
from qiskit.wrapper.credentials._configrc import (store_credentials,
                                                  get_qiskitrc_credentials)
from ._circuittoolkit import circuit_from_qasm_file, circuit_from_qasm_string

logger = logging.getLogger(__name__)


def register(token=None, url='https://quantumexperience.ng.bluemix.net/api',
             hub=None, group=None, project=None, proxies=None, verify=True,
             provider_name=None, provider_class=None, save_credentials=False):
    """Authenticate against an online backend provider.
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
        provider_class (BaseProvider): A Provider instance to be registered.
        save_credentials (bool): Store credentials in local qiskitrc
            file for later use. Default is False.
    Raises:
        QISKitError: if the provider name is not recognized.
    """
    if provider_class is not None:
        if not isinstance(provider_class, BaseProvider):
            raise QISKitError('provider_class must be a valid Provider instance.')
        else:
            reg._DEFAULT_PROVIDER.add_provider(provider_class)
            reg._DEFAULT_PROVIDER._find_backend_override_names()
            return

    did_register = 0
    if token is not None:
        url = _parse_ibmq_credentials(url, hub, group, project)
        reg._register(token, url, proxies, verify)
        reg.REGISTER_CALLED = 1
        did_register += 1
        if save_credentials:
            store_credentials(token=token, url=url,
                              proxies=proxies, verify=verify,
                              overwrite=False)
    else:
        specific_provider = False
        if provider_name is not None:
            specific_provider = True
        if not specific_provider:
            # Look for Qconfig.py in cwd
            did_register = reg.get_qconfig_credentials()
            # Look at env variables
            did_register = reg.get_env_credentials()
            # Look at qiksitrc for saved data
            did_register = get_qiskitrc_credentials()
            if not did_register:
                raise QISKitError(
                    "Registration failed: No provider credentials found.")
            reg.REGISTER_CALLED = 1
        else:
            did_register = get_qiskitrc_credentials(provider_name)
            if not did_register:
                raise QISKitError("Registration failed: Provider credentials not in qiskitrc file.")
            reg.REGISTER_CALLED = 1
        reg._DEFAULT_PROVIDER._find_backend_override_names()


def unregister(provider_name):
    """
    Removes a provider of list of registered providers.
    Args:
        provider_name (str): The unique name for the online provider.

    Raises:
        QISKitError: if the provider name is not recognized.
    """
    if provider_name == 'local':
        raise QISKitError("Cannot unregister 'local' provider.")
    else:
        _registered_names = [p.name for p in reg._DEFAULT_PROVIDER.providers]
        for idx, name in enumerate(_registered_names):
            if provider_name == name:
                reg._DEFAULT_PROVIDER.providers.pop(idx)
                reg._DEFAULT_PROVIDER._find_backend_override_names()
                return
        raise QISKitError('Provider %s is not registered.' % provider_name)


def registered_providers():
    """
    Returns the current list of available providers.

    Returns:
        list: Available providers.
    """
    return [p.name for p in reg._DEFAULT_PROVIDER.providers]


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
                     for backend in reg._DEFAULT_PROVIDER.available_backends(filters)]

    alias_dict = {v: k for k,
                  v in reg._DEFAULT_PROVIDER.aliased_backend_names().items()}
    backend_names = [alias_dict[name] if name in alias_dict else name for name in backend_names]

    if compact:
        group_dict = reg._DEFAULT_PROVIDER.grouped_backend_names()
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
    return reg._DEFAULT_PROVIDER.get_backend(name)


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
        backend = reg._DEFAULT_PROVIDER.get_backend(backend)

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
        backend = reg._DEFAULT_PROVIDER.get_backend(backend)
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


def qobj_to_circuits(qobj):
    """Return a list of QuantumCircuit object(s) from a qobj

    Args:
        qobj (Qobj): The Qobj object to convert to QuantumCircuits
    Returns:
        list: A list of QuantumCircuit objects from the qobj

    """
    if qobj.experiments:
        circuits = []
        for x in qobj.experiments:
            if hasattr(x.header, 'compiled_circuit_qasm'):
                circuits.append(
                    load_qasm_string(x.header.compiled_circuit_qasm))
        return circuits
    # TODO(mtreinish): add support for converting a qobj if the qasm isn't
    # embedded in the header
    return None
