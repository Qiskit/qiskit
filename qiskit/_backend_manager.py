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
import importlib
import inspect
import logging
import os
import pkgutil
from collections import namedtuple
from types import ModuleType

import qiskit
import qiskit.backends
from qiskit.backends import BaseBackend
from qiskit import QISKitError

logger = logging.getLogger(__name__)

RegisteredBackend = namedtuple('RegisteredBackend',
                               ['name', 'cls', 'configuration'])

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
        

def register(token, url='https://quantumexperience.ng.bluemix.net/api',
             hub=None, group=None, project=None, package=qiskit):
    """Register backends from the specified package.

    By calling this method, all available backends from this package
    are registered into QISKit.

    Args:
        token (str): user authentication token
        url (str): the url to the API
        hub (str): optional user hub
        group (str): optional user group
        project (str): optional user project
        package (ModuleType): the package to register backends from. This
             package will be checked for classes which inherit from BaseBackend.
    Returns:
        : list of names of the backends successfully registered
    """
    config = {
        'url': url,
        'hub': hub,
        'group': group,
        'project': project
    }
    # setup credential configuration parameters for registering backends
    # this configuration will be merged in to the one supplied by the backend.
    configuration = {'credentials': {'token': token,
                                     'config': config}}
    return discover_backend_classes(package,
                                    configuration=configuration)
    # from IBMQuantumExperience import IBMQuantumExperience
    # api_temp = IBMQuantumExperience(token, config)
    # api = API(api_temp)
    # qiskit.backends.discover_remote_backends(api_temp)

    # ally this would make an API object based on url and the user token
    # and register all the backends of this API to qiskit.backends.remote()
    # I am worried that there is not checks to see if the backends have the same name
    # this should be verified in the future.the input to the discover_remote_backends 
    # should be the api not api_temp

    # Ide

def discover_backend_classes(package, configuration=None):
    """This function attempts to discover all backend classes in the specified
    package.

    Args:
        package (module): module to search for classes derived from BaseBackend

    Returns:
        : list of backend names successfully registered
    Raises:
        TypeError: package is not a module
    """
    if not isinstance(package, ModuleType):
        raise TypeError('Argument "package" should be a module. '
                        'Received {}'.format(type(package)))
    backend_name_list = []
    for _, name, _ in pkgutil.walk_packages(package.__file__):
        # not sure why test directory is getting walked
        if 'test.python' in name:
            continue
        modspec = importlib.util.find_spec(name)
        try:
            mod = importlib.util.module_from_spec(modspec)
            modspec.loader.exec_module(mod)
        except Exception as err:
            logger.info('error checking for backend in {}'.format(
                name))
            continue
        for _, cls in inspect.getmembers(mod, inspect.isclass):
            # Iterate through the classes defined on the module.
            if (issubclass(cls, BaseBackend) and
                    cls.__module__ == modspec.name):
                try:
                    backend_name = register_backend(cls, configuration=configuration)
                    importlib.import_module(name)
                    backend_name_list.extend(backend_name if isinstance(
                        backend_name, list) else [backend_name])
                except QISKitError:
                    # Ignore backends that could not be initialized.
                    logger.info(
                        'backend %s could not be initialized', name)
    return backend_name_list

def register_backend(cls, configuration=None):
    """Register a backend in the list of available backends.

    Register a `cls` backend in the `_REGISTERED_BACKENDS` dict, validating
    that:
    * it follows the `BaseBackend` specification.
    * it can instantiated in the current context.
    * the backend is not already registered.

    Args:
        cls (class): a subclass of BaseBackend that contains a backend
        configuration (dict): backend configuration to use instead of class'
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

    # check to see if class provides configurations.
    available_backends = getattr(cls, 'available_backends', None)
    if callable(available_backends):
        available_configurations = available_backends(configuration=configuration)
    else:
        available_configurations = [configuration]
    if available_configurations == []:
        available_configurations = [None]
    backend_name_list = []
    for conf in available_configurations:
        try:
            backend_instance = cls(configuration=conf)
        except Exception as err:
            raise QISKitError('Could not register backend: %s could not be '
                              'instantiated: %s' % (cls, err))

        # Verify that it has a minimal valid configuration.
        try:
            backend_name = backend_instance.configuration['name']
        except (LookupError, TypeError):
            raise QISKitError('Could not register backend: invalid configuration')

        # insert backend reference
        if backend_name in _REGISTERED_BACKENDS:

            raise QISKitError(
                'backend name "{}" has already been registered'.format(
                    backend_name))
        registered_backend = RegisteredBackend(
            backend_name, cls, backend_instance.configuration)
        _REGISTERED_BACKENDS[backend_name] = registered_backend
        backend_name_list.append(backend_name)
    if len(backend_name_list) == 1:
        return backend_name_list[0]
    elif len(backend_name_list) > 1:
        return backend_name_list
    else:
        raise QISKitError('Could not register backend for this class')

def local_backends():
    """Get the local backends."""
    return [backend.name for backend in _REGISTERED_BACKENDS.values()
            if backend.configuration.get('local') is True]


def remote_backends():
    """Get the remote backends."""
    return [backend.name for backend in _REGISTERED_BACKENDS.values()
            if backend.configuration.get('local') is False]

def available_backends():
    """Get all available backend names."""
    return [backend.name for backend in _REGISTERED_BACKENDS.values()]

def get_backend(backend_name):
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
    
# class API(object):
#     """Creates an API object."""

#     # Functions to add
#     #   status -- gives the status of the api
#     #   available_backends -- all backends in this API, with their config
#     # A use case is the user would do
#     # ibmqx = qiskit.api.register(token,url)
#     # ibmqx.status and it prints the current status of the API

#     def __init__(self, api):
#         """Create an API object."""

#         # Ideally we should give this a url, but while we import the IBMQuantumExperience object
#         # I think this the best until we bring functions from IBMQuantumExperience into this object
#         self._api = api

#     def available_backends(self):
#         """Returns the backends on the api"""
#         return self._api.available_backends()
