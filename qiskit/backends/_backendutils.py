import os
import pkgutil
import importlib
import inspect
import sys
from qiskit.backends._basebackend import BaseBackend

# This dict holds '<backend name>': <backend class object> records and
# is imported to package scope.
_backend_classes = {}
_backend_configurations = {}

def update_implemented_backends():
    """This function attempts to discover all backend modules.

    Backend modules should subclass BaseBackend. Alternatively they need
    to define a module level __configuration dictionary and a class which
    implements a run() method.

    Returns:
        dict of '<backend name>': <backend class object>
    """
    for mod_info, name, ispkg in pkgutil.iter_modules([os.path.dirname(__file__)]):
        if name not in __name__:  # skip this module
            fullname = os.path.splitext(__name__)[0] + '.' + name
            modspec = importlib.util.find_spec(fullname)
            mod = importlib.util.module_from_spec(modspec)
            modspec.loader.exec_module(mod)
            if hasattr(mod, '__configuration'):
                _backend_configurations[mod.__configuration['name']] = mod.__configuration
                for class_name, class_obj in inspect.getmembers(mod,
                                                                inspect.isclass):
                    if hasattr(class_obj, 'run'):
                        class_obj = getattr(mod, class_name)
                        _backend_classes[mod.__configuration['name']] = class_obj
                        importlib.import_module(fullname)
            else:
                for class_name, class_obj in inspect.getmembers(
                        mod, inspect.isclass):
                    if issubclass(class_obj, BaseBackend):
                        try:
                            instance = class_obj({})
                        except:
                            instance = None
                        if isinstance(instance, BaseBackend):
                            backend_name = instance.configuration['name']
                            _backend_classes[backend_name] = class_obj
                            _backend_configurations[backend_name] = instance.configuration
                            importlib.import_module(fullname)
    return _backend_classes

def find_runnable_backends(backend_classes):
    backend_list = []
    circuit = {'header': {'clbit_labels': [['cr', 1]],
                          'number_of_clbits': 1,
                          'number_of_qubits': 1,
                          'qubit_labels': [['qr', 0]]},
               'operations':
                   [{'name': 'h',
                     'params': [],
                     'qubits': [0]},
                    {'clbits': [0],
                     'name': 'measure',
                     'qubits': [0]}]}
    qobj = {'id': 'backend_discovery',
            'config': {
                'max_credits': 3,
                'shots': 1,
                'backend': None,
                },
            'circuits': [circuit]
           }
    for backend_id, backend in _backend_classes.items():
        try:
            backend(qobj)
        except FileNotFoundError as fnferr:
            # this is for discovery so just don't add to discovered list
            pass
        else:
            backend_list.append(backend_id)
    return backend_list

def get_backend_class(backend_name):
    """Return the class object for the named backend.

    Args:
        backend_name (str): the backend name
    
    Returns:
        class object for backend_name

    Raises:
        LookupError if backend is unavailable
    """
    if backend_name in _backend_classes:
        return _backend_classes[backend_name]
    else:
        raise LookupError('backend "{}" is not available'.format(backend_name))

def get_backend_configuration(backend_name):
    """Return the configuration for the named backend.

    Args:
        backend_name (str): the backend name

    Returns:
        configuration dict

    Raises:
        LookupError if backend is unavailable
    """
    if backend_name in _backend_configurations:
        return _backend_configurations[backend_name]
    else:
        raise LookupError('backend "{}" is not available'.format(backend_name))
    
def local_backends():
    """Get the local backends."""
    local_backends = []
    for backend in _backend_configurations:
        configuration = get_backend_configuration(backend)
        # can drop this check once qobj works for remote
        if 'local' in configuration:
            if configuration['local'] == True:
                local_backends.append(backend)
    return local_backends

def remote_backends():
    """Get the remote backends."""
    remote_backends = []
    for backend in _backend_configurations:
        configuration = get_backend_configuration(backend)
        # can drop this check once qobj works for remote
        if 'local' in configuration:
            if configuration['local'] == False:
                remote_backends.append(backend)
    return remote_backends

