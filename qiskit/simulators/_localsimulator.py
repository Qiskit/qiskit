"""Interface to local simulators.

This module is the interface to all the local simulators in this directory.
It handles automatically discovering and interfacing with those modules. Once
instantiated like::

>>> import _localsimulator
>>> simulator_list = _localsimulator.local_backends()
>>> sim = _localsimulator.LocalSimulator(simulator_list[0], job)
>>> sim.run()
>>> results = sim.results()

`simulator_list` is the list of names of known simulators and `job` is
a dictionary of the form {'compiled_circuit': circuit, 'shots': shots,
'seed': seed}.

The import does discovery of the simulator modules in this directory. The
second command attempts to determine which modules are functional, in
particular for modules which require making calls to compiled binaries.

In order for a module to be registered it needs to define a module-scope
dictionary of the form::

    __configuration = {'name': 'local_qasm_simulator',
                       'url': 'https://github.com/IBM/qiskit-sdk-py',
                       'simulator': True,
                       'description': 'A python simulator for qasm files',
                       'coupling_map': 'all-to-all',
                       'basis_gates': 'u1,u2,u3,cx,id'}

and it needs a class with a "run" method. The identifier for the backend
simulator comes from the "name" key in this dictionary. The class'
__init__ method is called with a single `job` argument. The __init__
method is also responsible for determining whether an associated
binary is available. If it is not, the FileNotFoundError exception
should be raised.

Attributes:
    local_configuration : list of dict()
        This list gets populated with the __configuration records from each
        of the discovered modules.

    _simulator_classes : dict {"<simulator name>" : <simulator class>}
        This dictionary associates a simulator name with the class which
        generates its objects.
"""
import os
import pkgutil
import importlib
import inspect
import json

local_configuration = []
_simulator_classes = {}
for mod_info, name, ispkg in pkgutil.iter_modules([os.path.dirname(__file__)]):
    if name not in __name__:  # skip this module
        fullname = os.path.splitext(__name__)[0] + '.' + name
        modspec = importlib.util.find_spec(fullname)
        mod = importlib.util.module_from_spec(modspec)
        modspec.loader.exec_module(mod)
        if hasattr(mod, '__configuration'):
            local_configuration.append(mod.__configuration)
            for class_name, class_obj in inspect.getmembers(mod,
                                                            inspect.isclass):
                if hasattr(class_obj, 'run'):
                    class_obj = getattr(mod, class_name)
                    _simulator_classes[mod.__configuration['name']] = class_obj


def local_backends():
    """
    Attempt to determine the local simulator backends which are available.

    Mostly we are concerned about checking for whether the simulator executable
    associated with the discovered backend module exists. Discovery is done
    by instantiating the simulator object which should raise a
    FileNotFoundError if the program has not been compiled and placed in
    the executable path.

    Returns:
        A list of backend names.
    """
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
    job = {'compiled_circuit': json.dumps(circuit).encode(),
           'config': {'shots': 1, 'seed': None}
           }
    for backend_id, backend in _simulator_classes.items():
        try:
            sim = backend(job)
        except FileNotFoundError as fnferr:
            # this is for discovery so just don't had to discovered list
            pass
        else:
            backend_list.append(backend_id)
    return backend_list


class LocalSimulator:
    """
    Interface to simulators
    """
    def __init__(self, backend, job):
        self._backend = backend
        self._job = job
        self._result = {'data': None, 'status': "Error"}
        self._sim = _simulator_classes[backend](job)

    def run(self, silent=False):
        simOutput = self._sim.run(silent)
        self._result["data"] = simOutput["data"]
        self._result["status"] = simOutput["status"]

    def result(self):
        return self._result
