from collections import namedtuple
import importlib
import inspect
import os
import pkgutil
import logging

from .. import QISKitError
from ._basebackend import BaseBackend

logger = logging.getLogger(__name__)

RegisteredBackend = namedtuple('RegisteredBackend',
                               ['name', 'cls', 'configuration'])

_REGISTERED_BACKENDS = {}
"""dict (backend_name: RegisteredBackend) with the available backends.

Dict that contains the available backends during the current invocation of the
SDK, with the form `'<backend name>': <RegisteredBackend object>`.

Please note that this variable will not contain the full list until runtime,
as its contents are a combination of:
* backends that are auto-discovered by :func:`discover_sdk_backend`, as
  they might have external dependencies or not be part of the SDK standard
  backends.
* backends registered manually by the user by :func:`register_backend`.
"""


def discover_local_backends(directory=os.path.dirname(__file__)):
    """This function attempts to discover all backend modules.

    Discover the backends on modules on the directory of the current module
    and attempt to register them. Backend modules should subclass BaseBackend.

    Args:
        directory (str, optional): Directory to search for backends. Defaults
            to the directory of this module.
    """
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
                        register_backend(cls)
                        importlib.import_module(fullname)
                    except QISKitError:
                        # Ignore backends that could not be initialized.
                        logger.info(
                            'backend {} could not be initialized'.format(
                                fullname))

def discover_remote_backends(api):
    """Discover backends available on the Quantum Experience

    Args:
        api (IBMQuantumExperience): Quantum Experience API
    """
    from ._qeremote import QeRemote
    QeRemote.set_api(api)
    configuration_list = api.available_backends()
    for configuration in configuration_list:
        backend_name = configuration['name']
        configuration['local'] = False
        registered_backend = RegisteredBackend(backend_name,
                                               QeRemote,
                                               configuration)
        _REGISTERED_BACKENDS[backend_name] = registered_backend

def register_backend(cls):
    """Register a backend in the list of available backends.

    Register a `cls` backend in the `_REGISTERED_BACKENDS` dict, validating
    that:
    * it follows the `BaseBackend` specification.
    * it can instantiated in the current context.
    * the backend is not already registered.

    Args:
        cls (BaseBackend): a subclass of BaseBackend that contains a backend

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
        backend_instance = cls()
    except Exception as e:
        raise QISKitError('Could not register backend: %s could not be '
                          'instantiated: %s' % (cls, e))

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


def get_backend_class(backend_name):
    """Return the class object for the named backend.

    Args:
        backend_name (str): the backend name

    Returns:
        class object for backend_name

    Raises:
        LookupError if backend is unavailable
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
        instance subclass of BaseBackend

    Raises:
        LookupError if backend is unavailable
    """
    try:
        registered_backend = _REGISTERED_BACKENDS[backend_name]
    except KeyError:
        raise LookupError('backend "{}" is not available'.format(backend_name))
    return registered_backend.cls(configuration=registered_backend.configuration)

def get_backend_configuration(backend_name):
    """Return the configuration for the named backend.

    Args:
        backend_name (str): the backend name

    Returns:
        configuration dict

    Raises:
        LookupError if backend is unavailable
    """
    try:
        return _REGISTERED_BACKENDS[backend_name].configuration
    except KeyError:
        raise LookupError('backend "{}" is not available'.format(backend_name))


def local_backends():
    """Get the local backends."""
    return [backend.name for backend in _REGISTERED_BACKENDS.values()
            if backend.configuration.get('local') == True]


def remote_backends():
    """Get the remote backends."""
    return [backend.name for backend in _REGISTERED_BACKENDS.values()
            if backend.configuration.get('local') == False]


discover_local_backends()
