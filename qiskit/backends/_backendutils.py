# -*- coding: utf-8 -*-

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

"""Backend functions for registration, information, etc."""

import importlib
import inspect
import logging
import os
import pkgutil
import re
from collections import namedtuple
from itertools import combinations

import qiskit
from ._basebackend import BaseBackend
from .. import QISKitError

logger = logging.getLogger(__name__)

RegisteredBackend = namedtuple('RegisteredBackend',
                               ['name', 'cls', 'configuration'])

_REGISTERED_BACKENDS = {}
"""dict (backend_name: RegisteredBackend) with the available backends.

Dict that contains the available backends during the current invocation of the
SDK, with the form `'<backend name>': <RegisteredBackend object>`.

Please note that this variable will not contain the full list until runtime,
as its contents are a combination of:
* backends that are auto-discovered by :func:`discover_local_backends` and
  :func:`discover_remote_backends`, as they might have external dependencies
  or not be part of the SDK standard backends.
* backends registered manually by the user by :func:`register_backend`.
"""

_ALIASED_BACKENDS = {
        'local_qasm_simulator': ['local_qasm_simulator_cpp',
                                 'local_qasm_simulator_projectq',
                                 'local_qasm_simulator_py'],
        'local_statevector_simulator': ['local_statevector_simulator_cpp',
                                        'local_statevector_simulator_projectq',
                                        'local_statevector_simulator_py',
                                        'local_statevector_simulator_sympy'],
        'local_unitary_simulator': ['local_unitary_simulator_cpp',
                                    'local_unitary_simulator_py',
                                    'local_unitary_simulator_sympy'],
        'local_clifford_simulator': ['local_clifford_simulator_cpp'],
        'ibmq_qasm_simulator': ['ibmq_qasm_simulator',
                                'ibmq_qasm_simulator_hpc']
        }
"""
dict (alias_name: backend_names(list))

Dict that defines alias names, usually shorter names for referring to
the backends.

If an alias key is used, the corresponding backend will be chosen in order
of priority from the value list, depending on availability.
"""

_DEPRECATED_BACKENDS = {
        'local_qiskit_simulator': 'local_qasm_simulator_cpp',
        'wood_simulator': 'local_qasm_simulator_cpp',
        'real': 'ibmqx1',
        'ibmqx_qasm_simulator': 'ibmq_qasm_simulator',
        'ibmqx_hpc_qasm_simulator': 'ibmq_qasm_simulator_hpc'
        }
"""
dict (deprecated_name: backend_name)

Dict that stores the current name for all deprecated backends.
These will be removed in future releases.
"""

FIRST_CAP_RE = re.compile('(.)([A-Z][a-z]+)')
ALL_CAP_RE = re.compile('([a-z0-9])([A-Z])')


def discover_local_backends(directory=os.path.dirname(__file__)):
    """This function attempts to discover all backend modules.

    Discover the backends on modules on the directory of the current module
    and attempt to register them. Backend modules should subclass BaseBackend.

    Args:
        directory (str, optional): Directory to search for backends. Defaults
            to the directory of this module.

    Returns:
        list: list of backend names discovered
    """
    # check aliases have been defined correctly (max one alias per backend)
    for pair in combinations(_ALIASED_BACKENDS.values(), r=2):
        if not set.isdisjoint(set(pair[0]), set(pair[1])):
                raise ValueError('duplicate backend alias definition')

    # discover the local backends
    backend_name_list = []
    for _, name, _ in pkgutil.iter_modules([directory]):
        # Iterate through the modules on the directory of the current one.
        if name not in __name__:  # skip the current module
            fullname = os.path.splitext(__name__)[0] + '.' + name
            modspec = importlib.util.find_spec(fullname)
            mod = importlib.util.module_from_spec(modspec)
            modspec.loader.exec_module(mod)
            for _, cls in inspect.getmembers(mod, inspect.isclass):
                # Iterate through the classes defined on the module.
                if (issubclass(cls, BaseBackend) and
                        cls.__module__ == modspec.name):
                    try:
                        backend_name = register_backend(cls)
                        backend_name_list.append(backend_name)
                        importlib.import_module(fullname)
                    except QISKitError:
                        # Ignore backends that could not be initialized.
                        logger.info(
                            'backend %s could not be initialized', fullname)
    return backend_name_list


def discover_remote_backends(api):
    """Discover backends available from IBM Q

    Args:
        api (IBMQuantumExperience): IBM Q API
    Returns:
        list: list of discovered backend names
    """
    from ._qeremote import QeRemote
    QeRemote.set_api(api)
    config_list = api.available_backends()
    backend_name_list = []
    for config in config_list:
        config_edit = {}
        backend_name = config['name']
        backend_name_list.append(backend_name)
        config_edit['local'] = False
        for key in config.keys():
            new_key = _snake_case_to_camel_case(key)
            if new_key not in ['id', 'serial_number', 'topology_id', 'status']:
                config_edit[new_key] = config[key]
        # online_qasm_simulator uses different name for basis_gates
        if 'gateSet' in config:
            config_edit['basis_gates'] = config['gateSet']
            del config_edit['gate_set']
        # ibmqx_qasm_simulator doesn't report coupling_map
        if 'coupling_map' not in config_edit.keys() and config['simulator']:
            config_edit['coupling_map'] = 'all-to-all'
        registered_backend = RegisteredBackend(backend_name,
                                               QeRemote,
                                               config_edit)
        _REGISTERED_BACKENDS[backend_name] = registered_backend
    return backend_name_list


def _snake_case_to_camel_case(name):
    """Return a snake case string from a camelcase string."""
    string_1 = FIRST_CAP_RE.sub(r'\1_\2', name)
    return ALL_CAP_RE.sub(r'\1_\2', string_1).lower()

def update_backends(api=None):
    """Update registered backends.

    This function deletes refreshes list of available backends

    Args:
        api (IBMQuantumExperience): api to use to check for backends.

    Returns:
        list: list of discovered backend names
    """
    _REGISTERED_BACKENDS.clear()
    backend_name_list = []
    backend_name_list += discover_local_backends()
    if api is not None:
        backend_name_list += discover_remote_backends(api)
    return backend_name_list


def register_backend(cls, configuration_=None):
    """Register a backend in the list of available backends.

    Register a `cls` backend in the `_REGISTERED_BACKENDS` dict, validating
    that:
    * it follows the `BaseBackend` specification.
    * it can instantiated in the current context.
    * the backend is not already registered.

    Args:
        cls (class): a subclass of BaseBackend that contains a backend
        configuration_ (dict): backend configuration to use instead of class'
            default.

    Returns:
        string: the identifier of the backend

    Raises:
        QISKitError: if `cls` is not a valid Backend.
    """

    # Verify that the backend is not already registered.
    if cls in [backend.cls for backend in _REGISTERED_BACKENDS.values()]:
        raise QISKitError('Could not register backend: %s is not a subclass '
                          'of BaseBackend' % cls)

    # Verify that it is a subclass of BaseBackend.
    if not issubclass(cls, BaseBackend):
        raise QISKitError('Could not register backend: %s is not a subclass '
                          'of BaseBackend' % cls)
    try:
        backend_instance = cls(configuration=configuration_)
    except Exception as err:
        raise QISKitError('Could not register backend: %s could not be '
                          'instantiated: %s' % (cls, err))

    # Verify that it has a minimal valid configuration.
    try:
        backend_name = backend_instance.configuration['name']
    except (LookupError, TypeError):
        raise QISKitError('Could not register backend: invalid configuration')

    # Append the backend to the `_backend_classes` dict.
    registered_backend = RegisteredBackend(
        backend_name, cls, backend_instance.configuration)
    _REGISTERED_BACKENDS[backend_name] = registered_backend

    return backend_name


def deregister_backend(backend_name):
    """Remove backend from list of available backens

    Args:
        backend_name (str): name of backend to deregister

    Raises:
        KeyError if backend_name is not registered.
    """
    _REGISTERED_BACKENDS.pop(backend_name)


def get_backend_class(backend_name):
    """Return the class object for the named backend.

    Args:
        backend_name (str): the backend name

    Returns:
        BaseBackend: class object for backend_name

    Raises:
        LookupError: if backend is unavailable
    """
    try:
        return _REGISTERED_BACKENDS[backend_name].cls
    except KeyError:
        raise LookupError('backend "{}" is not available'.format(backend_name))


def get_backend_instance(backend_name):
    """Return a backend instance for the named backend.

    Args:
        backend_name (str): the backend name

    Returns:
        BaseBackend: instance subclass of BaseBackend

    Raises:
        LookupError: if backend is unavailable
    """
    try:
        backend_name = resolve_name(backend_name)
        registered_backend = _REGISTERED_BACKENDS[backend_name]
        return registered_backend.cls(
            configuration=registered_backend.configuration)
    except KeyError:
        raise LookupError('backend "{}" is not available'.format(backend_name))


def configuration(backend_name):
    """Return the configuration for the named backend.

    Args:
        backend_name (str): the backend name

    Returns:
        dict: configuration dict

    Raises:
        LookupError: if backend is unavailable
    """
    try:
        backend = qiskit.backends.get_backend_instance(backend_name)        
        return backend.configuration
    except KeyError:
        raise LookupError('backend "{}" is not available'.format(backend_name))


def calibration(backend_name):
    """Return the calibration for the named backend.

    Args:
        backend_name (str): the backend name

    Returns:
        dict: calibration dict

    Raises:
        LookupError: if backend is unavailable
    """
    try:
        backend = qiskit.backends.get_backend_instance(backend_name)
        return backend.calibration
    except KeyError:
        raise LookupError('backend "{}" is not available'.format(backend_name))


def parameters(backend_name):
    """Return the online backend parameters.

    Args:
        backend_name (str):  Name of the backend.

    Returns:
        dict: The parameters of the named backend.

    Raises:
        ConnectionError: if the API call failed.
        LookupError: If parameters for the named backend can't be
            found.
    """
    try:
        backend = qiskit.backends.get_backend_instance(backend_name)
        return backend.parameters
    except KeyError:
        raise LookupError('backend "{}" is not available'.format(backend_name))


def status(backend_name):
    """Return the status for the named backend.

    Args:
        backend_name (str): the backend name

    Returns:
        dict: status dict

    Raises:
        LookupError: if backend is unavailable
    """
    try:
        backend = qiskit.backends.get_backend_instance(backend_name)
        return backend.status
    except KeyError:
        raise LookupError('backend "{}" is not available'.format(backend_name))


def local_backends(compact=True):
    """Get the local backends.
    
    Args:
        compact (bool): only report alias names. this is usually shorter, any several
        backends usually share the same alias.

    Returns:
        list(str): local backend names
    """
    registered_local = [backend.name for backend in _REGISTERED_BACKENDS.values()
                        if backend.configuration.get('local') is True]
    # if an alias has been defined, report that. otherwise use its own name
    if compact:
        aliases = set()
        for backend in registered_local:
            backend_alias = set(k for k, v in _ALIASED_BACKENDS.items() if backend in v)
            if len(backend_alias) == 0:
                aliases.add(backend)
            elif len(backend_alias) == 1:
                (alias,) = backend_alias
                aliases.add(alias)
        registered_local = list(aliases)

    return registered_local


def remote_backends(compact=False):
    """Get the remote backends.

    Args:
        compact (bool): only report alias names. this is usually shorter, any several
        backends usually share the same alias.

    Returns:
        list(str): remote backend names
    """
    registered_remote = [backend.name for backend in _REGISTERED_BACKENDS.values()
                        if backend.configuration.get('local') is False]
    # if an alias has been defined, report that. otherwise use its own name
    if compact:
        aliases = set()
        for backend in registered_remote:
            backend_alias = set(k for k, v in _ALIASED_BACKENDS.items() if backend in v)
            if len(backend_alias) == 0:
                aliases.add(backend)
            elif len(backend_alias) == 1:
                (alias,) = backend_alias
                aliases.add(alias)
        registered_remote = list(aliases)

    return registered_remote


def resolve_name(backend):
    """Resolve backend name from a possible short alias or a deprecated name.

    The alias will be chosen in order of priority, depending on availability.

    Args:
        backend (str): name of backend to resolve

    Returns:
        str: name of resolved backend, which exists in _REGISTERED_BACKENDS

    Raises:
        LookupError: if backend cannot be resolved through registered names,
        nor aliases, nor deprecated names
    """
    resolved_backend = ""
    if backend in _REGISTERED_BACKENDS:
        resolved_backend = backend
    elif backend in _ALIASED_BACKENDS:
        available_aliases = [b for b in _ALIASED_BACKENDS[backend]
                             if b in _REGISTERED_BACKENDS]
        if available_aliases:
            resolved_backend = available_aliases[0]
    elif backend in _DEPRECATED_BACKENDS:
            resolved_backend = _DEPRECATED_BACKENDS[backend]
            logger.warning('WARNING: "{0}" is deprecated. Use "{1}"'.format(
                            backend, resolved_backend))

    if resolved_backend not in _REGISTERED_BACKENDS:
        raise LookupError('backend "{}" is not available'.format(backend))

    return resolved_backend

discover_local_backends()
