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

import qiskit
from ._basebackend import BaseBackend
from .. import QISKitError

logger = logging.getLogger(__name__)

RegisteredBackend = namedtuple('RegisteredBackend',
                               ['name', 'cls', 'configuration', 'api'])

_REGISTERED_BACKENDS = {}
"""dict (backend_name: RegisteredBackend) with the available backends.

Dict that contains the available backends during the current invocation of the
SDK, with the form `'<backend name>': <RegisteredBackend object>`.

As part of the backend registration process, a record of the backend's
name, class, configuration, and api is kept.

Please note that this variable will not contain the full list until runtime,
as its contents are a combination of:
* backends that are auto-discovered by :func:`discover_sdk_backend`, as
  they might have external dependencies or not be part of the SDK standard
  backends.
* backends registered manually by the user by :func:`register_backend`.
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
                                               config_edit,
                                               api)
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


def register_backend(cls, configuration_=None, api=None):
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
        backend_name, cls, backend_instance.configuration, api)
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
        return _REGISTERED_BACKENDS[backend_name].configuration
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

def api(backend_name):
    """Return the api that the named backend belongs to.

    Args:
        backend_name (str): the backend name

    Returns:
        API: the api that the backend belongs to

    Raises:
        LookupError: if backend is unavailable
        RuntimeError: if a local backend has api other than None
    """
    try:
        backend = _REGISTERED_BACKENDS[backend_name]
        if backend.configuration.get('local') is True and backend.api is not None:
            raise RuntimeError('backend "{}" registered as local, '
                               'but points to a remote API')
        return backend.api
    except KeyError:
        raise LookupError('backend "{}" is not available'.format(backend_name))
    

def local_backends():
    """Get the local backends."""
    return [backend.name for backend in _REGISTERED_BACKENDS.values()
            if backend.configuration.get('local') is True]


def remote_backends():
    """Get the remote backends."""
    return [backend.name for backend in _REGISTERED_BACKENDS.values()
            if backend.configuration.get('local') is False]


discover_local_backends()
