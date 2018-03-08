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

"""
Qasm Program Class
"""

import random
import json
import logging
import os
import string
import re
from threading import Event
import copy

# use the external IBMQuantumExperience Library
from IBMQuantumExperience import IBMQuantumExperience

# Local Simulator Modules
import qiskit.backends

# Stable Modules
from . import QuantumRegister
from . import ClassicalRegister
from . import QuantumCircuit
from . import QISKitError
from . import JobProcessor
from . import QuantumJob
from ._logging import set_qiskit_logger, unset_qiskit_logger

# Beta Modules
from . import unroll
from . import qasm
from . import mapper

from . import _openquantumcompiler as openquantumcompiler

FIRST_CAP_RE = re.compile('(.)([A-Z][a-z]+)')
ALL_CAP_RE = re.compile('([a-z0-9])([A-Z])')

logger = logging.getLogger(__name__)


def convert(name):
    """Return a snake case string from a camelcase string."""
    string_1 = FIRST_CAP_RE.sub(r'\1_\2', name)
    return ALL_CAP_RE.sub(r'\1_\2', string_1).lower()


class QuantumProgram(object):
    """Quantum Program Class.

     Class internal properties.

     Elements that are not python identifiers or string constants are denoted
     by "--description (type)--". For example, a circuit's name is denoted by
     "--circuit name (string)--" and might have the value "teleport".

     Internal::

        __quantum_registers (list[dic]): An dictionary of quantum registers
            used in the quantum program.
            __quantum_registers =
                {
                --register name (string)--: QuantumRegister,
                }
        __classical_registers (list[dic]): An ordered list of classical
            registers used in the quantum program.
            __classical_registers =
                {
                --register name (string)--: ClassicalRegister,
                }
        __quantum_program (dic): An dictionary of quantum circuits
            __quantum_program =
                {
                --circuit name (string)--:  --circuit object --,
                }
        __init_circuit (obj): A quantum circuit object for the initial quantum
            circuit
        __ONLINE_BACKENDS (list[str]): A list of online backends
        __LOCAL_BACKENDS (list[str]): A list of local backends
     """
    # -- FUTURE IMPROVEMENTS --
    # TODO: for status results make ALL_CAPS (check) or some unified method
    # TODO: Jay: coupling_map, basis_gates will move into a config object
    # only exists once you set the api to use the online backends
    __api = {}
    __api_config = {}

    def __init__(self, specs=None):
        self.__quantum_registers = {}
        self.__classical_registers = {}
        self.__quantum_program = {}  # stores all the quantum programs
        self.__init_circuit = None  # stores the intial quantum circuit of the program
        self.__ONLINE_BACKENDS = []  # pylint: disable=invalid-name
        self.__LOCAL_BACKENDS = qiskit.backends.local_backends()  # pylint: disable=invalid-name
        self.mapper = mapper
        if specs:
            self.__init_specs(specs)

    def enable_logs(self, level=logging.INFO):
        """Enable the console output of the logging messages.

        Enable the output of logging messages (above level `level`) to the
        console, by configuring the `qiskit` logger accordingly.

        Params:
            level (int): minimum severity of the messages that are displayed.

        Note:
            This is a convenience method over the standard Python logging
            facilities, and modifies the configuration of the 'qiskit.*'
            loggers. If finer control over the logging configuration is needed,
            it is encouraged to bypass this method.
        """
        # Update the handlers and formatters.
        set_qiskit_logger()
        # Set the logger level.
        logging.getLogger('qiskit').setLevel(level)

    def disable_logs(self):
        """Disable the console output of the logging messages.

        Disable the output of logging messages (above level `level`) to the
        console, by removing the handlers from the `qiskit` logger.

        Note:
            This is a convenience method over the standard Python logging
            facilities, and modifies the configuration of the 'qiskit.*'
            loggers. If finer control over the logging configuration is needed,
            it is encouraged to bypass this method.
        """
        unset_qiskit_logger()

    ###############################################################
    # methods to initiate an build a quantum program
    ###############################################################

    def __init_specs(self, specs):
        """Populate the Quantum Program Object with initial Specs.

        Args:
            specs (dict):
                    Q_SPECS = {
                        "circuits": [{
                            "name": "Circuit",
                            "quantum_registers": [{
                                "name": "qr",
                                "size": 4
                            }],
                            "classical_registers": [{
                                "name": "cr",
                                "size": 4
                            }]
                        }],
        """
        quantumr = []
        classicalr = []
        if "circuits" in specs:
            for circuit in specs["circuits"]:
                quantumr = self.create_quantum_registers(
                    circuit["quantum_registers"])
                classicalr = self.create_classical_registers(
                    circuit["classical_registers"])
                self.create_circuit(name=circuit["name"], qregisters=quantumr,
                                    cregisters=classicalr)
        # TODO: Jay: I think we should return function handles for the
        # registers and circuit. So that we dont need to get them after we
        # create them with get_quantum_register etc

    def create_quantum_register(self, name, size):
        """Create a new Quantum Register.

        Args:
            name (str): the name of the quantum register
            size (int): the size of the quantum register

        Returns:
            QuantumRegister: internal reference to a quantum register in
                __quantum_registers

        Raises:
            QISKitError: if the register already exists in the program.
        """
        if name in self.__quantum_registers:
            if size != len(self.__quantum_registers[name]):
                raise QISKitError("Can't make this register: Already in"
                                  " program with different size")
            logger.info(">> quantum_register exists: %s %s", name, size)
        else:
            self.__quantum_registers[name] = QuantumRegister(name, size)
            logger.info(">> new quantum_register created: %s %s", name, size)
        return self.__quantum_registers[name]

    def destroy_quantum_register(self, name):
        """Destroy an existing Quantum Register.

        Args:
            name (str): the name of the quantum register

        Raises:
            QISKitError: if the register does not exist in the program.
        """
        if name not in self.__quantum_registers:
            raise QISKitError("Can't destroy this register: Not present")
        else:
            logger.info(">> quantum_register destroyed: %s", name)
            del self.__quantum_registers[name]

    def create_quantum_registers(self, register_array):
        """Create a new set of Quantum Registers based on a array of them.

        Args:
            register_array (list[dict]): An array of quantum registers in
                dictionary format::

                    "quantum_registers": [
                        {
                        "name": "qr",
                        "size": 4
                        },
                        ...
                    ]
        Returns:
            list(QuantumRegister): Array of quantum registers objects
        """
        new_registers = []
        for register in register_array:
            register = self.create_quantum_register(
                register["name"], register["size"])
            new_registers.append(register)
        return new_registers

    def destroy_quantum_registers(self, register_array):
        """Destroy a set of Quantum Registers based on a array of them.

        Args:
            register_array (list[dict]): An array of quantum registers in
                dictionary format::

                    "quantum_registers": [
                        {
                        "name": "qr",
                        },
                        ...
                    ]

                "size" may be a key for compatibility, but is ignored.
        """
        for register in register_array:
            self.destroy_quantum_register(register["name"])

    def create_classical_register(self, name, size):
        """Create a new Classical Register.

        Args:
            name (str): the name of the classical register
            size (int): the size of the classical register
        Returns:
            ClassicalRegister: internal reference to a classical register
                in __classical_registers

        Raises:
            QISKitError: if the register already exists in the program.
        """
        if name in self.__classical_registers:
            if size != len(self.__classical_registers[name]):
                raise QISKitError("Can't make this register: Already in"
                                  " program with different size")
            logger.info(">> classical register exists: %s %s", name, size)
        else:
            logger.info(">> new classical register created: %s %s", name, size)
            self.__classical_registers[name] = ClassicalRegister(name, size)
        return self.__classical_registers[name]

    def create_classical_registers(self, registers_array):
        """Create a new set of Classical Registers based on a array of them.

        Args:
            registers_array (list[dict]): An array of classical registers in
                dictionary format::

                    "classical_registers": [
                        {
                        "name": "qr",
                        "size": 4
                        },
                        ...
                    ]
        Returns:
            list(ClassicalRegister): Array of clasical registers objects
        """
        new_registers = []
        for register in registers_array:
            new_registers.append(self.create_classical_register(
                register["name"], register["size"]))
        return new_registers

    def destroy_classical_register(self, name):
        """Destroy an existing Classical Register.

        Args:
            name (str): the name of the classical register

        Raises:
            QISKitError: if the register does not exist in the program.
        """
        if name not in self.__classical_registers:
            raise QISKitError("Can't destroy this register: Not present")
        else:
            logger.info(">> classical register destroyed: %s", name)
            del self.__classical_registers[name]

    def destroy_classical_registers(self, registers_array):
        """Destroy a set of Classical Registers based on a array of them.

        Args:
            registers_array (list[dict]): An array of classical registers in
                dictionary format::

                    "classical_registers": [
                        {
                        "name": "qr",
                        },
                        ...
                    ]

                "size" may be a key for compatibility, but is ignored.
        """
        for register in registers_array:
            self.destroy_classical_register(register["name"])

    def create_circuit(self, name, qregisters=None, cregisters=None):
        """Create a empty Quantum Circuit in the Quantum Program.

        Args:
            name (str): the name of the circuit.
            qregisters (list(QuantumRegister)): is an Array of Quantum
                Registers by object reference
            cregisters (list(ClassicalRegister)): is an Array of Classical
                Registers by object reference

        Returns:
            QuantumCircuit: A quantum circuit is created and added to the
                Quantum Program
        """
        if not qregisters:
            qregisters = []
        if not cregisters:
            cregisters = []
        quantum_circuit = QuantumCircuit()
        if not self.__init_circuit:
            self.__init_circuit = quantum_circuit
        for register in qregisters:
            quantum_circuit.add(register)
        for register in cregisters:
            quantum_circuit.add(register)
        self.add_circuit(name, quantum_circuit)
        return self.__quantum_program[name]

    def destroy_circuit(self, name):
        """Destroy a Quantum Circuit in the Quantum Program. This will not
        destroy any registers associated with the circuit.

        Args:
            name (str): the name of the circuit

        Raises:
            QISKitError: if the register does not exist in the program.
        """
        if name not in self.__quantum_program:
            raise QISKitError("Can't destroy this circuit: Not present")
        del self.__quantum_program[name]

    def add_circuit(self, name, quantum_circuit):
        """Add a new circuit based on an Object representation.

        Args:
            name (str): the name of the circuit to add.
            quantum_circuit (QuantumCircuit): a quantum circuit to add to the
                program-name
        """
        for qname, qreg in quantum_circuit.get_qregs().items():
            self.create_quantum_register(qname, len(qreg))
        for cname, creg in quantum_circuit.get_cregs().items():
            self.create_classical_register(cname, len(creg))
        self.__quantum_program[name] = quantum_circuit

    def load_qasm_file(self, qasm_file, name=None,
                       basis_gates='u1,u2,u3,cx,id'):
        """ Load qasm file into the quantum program.

        Args:
            qasm_file (str): a string for the filename including its location.
            name (str or None): the name of the quantum circuit after
                loading qasm text into it. If no name is give the name is of
                the text file.
            basis_gates (str): basis gates for the quantum circuit.
        Returns:
            str: Adds a quantum circuit with the gates given in the qasm file to the
            quantum program and returns the name to be used to get this circuit
        Raises:
            QISKitError: if the file cannot be read.
        """
        if not os.path.exists(qasm_file):
            raise QISKitError('qasm file "{0}" not found'.format(qasm_file))
        if not name:
            name = os.path.splitext(os.path.basename(qasm_file))[0]
        node_circuit = qasm.Qasm(filename=qasm_file).parse()  # Node (AST)
        logger.info("circuit name: %s", name)
        logger.info("******************************")
        logger.info(node_circuit.qasm())
        # current method to turn it a DAG quantum circuit.
        unrolled_circuit = unroll.Unroller(node_circuit,
                                           unroll.CircuitBackend(basis_gates.split(",")))
        circuit_unrolled = unrolled_circuit.execute()
        self.add_circuit(name, circuit_unrolled)
        return name

    def load_qasm_text(self, qasm_string, name=None,
                       basis_gates='u1,u2,u3,cx,id'):
        """ Load qasm string in the quantum program.

        Args:
            qasm_string (str): a string for the file name.
            name (str or None): the name of the quantum circuit after loading qasm
                text into it. If no name is give the name is of the text file.
            basis_gates (str): basis gates for the quantum circuit.
        Returns:
            str: Adds a quantum circuit with the gates given in the qasm string to
            the quantum program.
        """
        node_circuit = qasm.Qasm(data=qasm_string).parse()  # Node (AST)
        if not name:
            # Get a random name if none is given
            name = "".join([random.choice(string.ascii_letters+string.digits)
                            for n in range(10)])
        logger.info("circuit name: %s", name)
        logger.info("******************************")
        logger.info(node_circuit.qasm())
        # current method to turn it a DAG quantum circuit.
        unrolled_circuit = unroll.Unroller(node_circuit,
                                           unroll.CircuitBackend(basis_gates.split(",")))
        circuit_unrolled = unrolled_circuit.execute()
        self.add_circuit(name, circuit_unrolled)
        return name

    ###############################################################
    # methods to get elements from a QuantumProgram
    ###############################################################

    def get_quantum_register(self, name):
        """Return a Quantum Register by name.

        Args:
            name (str): the name of the quantum register
        Returns:
            QuantumRegister: The quantum register with this name
        Raises:
            KeyError: if the quantum register is not on the quantum program.
        """
        try:
            return self.__quantum_registers[name]
        except KeyError:
            raise KeyError('No quantum register "{0}"'.format(name))

    def get_classical_register(self, name):
        """Return a Classical Register by name.

        Args:
            name (str): the name of the classical register
        Returns:
            ClassicalRegister: The classical register with this name
        Raises:
            KeyError: if the classical register is not on the quantum program.
        """
        try:
            return self.__classical_registers[name]
        except KeyError:
            raise KeyError('No classical register "{0}"'.format(name))

    def get_quantum_register_names(self):
        """Return all the names of the quantum Registers."""
        return self.__quantum_registers.keys()

    def get_classical_register_names(self):
        """Return all the names of the classical Registers."""
        return self.__classical_registers.keys()

    def get_circuit(self, name):
        """Return a Circuit Object by name
        Args:
            name (str): the name of the quantum circuit
        Returns:
            QuantumCircuit: The quantum circuit with this name
        Raises:
            KeyError: if the circuit is not on the quantum program.
        """
        try:
            return self.__quantum_program[name]
        except KeyError:
            raise KeyError('No quantum circuit "{0}"'.format(name))

    def get_circuit_names(self):
        """Return all the names of the quantum circuits."""
        return self.__quantum_program.keys()

    def get_qasm(self, name):
        """Get qasm format of circuit by name.

        Args:
            name (str): name of the circuit

        Returns:
            str: The quantum circuit in qasm format
        """
        quantum_circuit = self.get_circuit(name)
        return quantum_circuit.qasm()

    def get_qasms(self, list_circuit_name):
        """Get qasm format of circuit by list of names.

        Args:
            list_circuit_name (list[str]): names of the circuit

        Returns:
            list(QuantumCircuit): List of quantum circuit in qasm format
        """
        qasm_source = []
        for name in list_circuit_name:
            qasm_source.append(self.get_qasm(name))
        return qasm_source

    def get_initial_circuit(self):
        """Return the initialization Circuit."""
        return self.__init_circuit

    ###############################################################
    # methods for working with backends
    ###############################################################

    def set_api(self, token, url, hub=None, group=None, project=None,
                proxies=None, verify=True):
        """ Setup the API.

        Fills the __ONLINE_BACKENDS, __api, and __api_config variables.
        Does not catch exceptions from IBMQuantumExperience.

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
        Raises:
            ConnectionError: if the API instantiation failed.
            QISKitError: if no hub, group or project were specified.
        """
        try:
            config_dict = {
                'url': url,
                'hub': hub,
                'group': group,
                'project': project
            }
            if proxies:
                config_dict['proxies'] = proxies
            self.__api = IBMQuantumExperience(token, config_dict, verify)
        except Exception as ex:
            root_exception = ex
            if 'License required' in str(ex):
                # For the 401 License required exception from the API, be
                # less verbose with the exceptions.
                root_exception = None
            raise ConnectionError("Couldn't connect to IBMQuantumExperience server: {0}"
                                  .format(ex)) from root_exception
        qiskit.backends.discover_remote_backends(self.__api)
        self.__ONLINE_BACKENDS = self.online_backends()
        self.__api_config["token"] = token
        self.__api_config["config"] = config_dict.copy()

    def set_api_hubs_config(self, hub, group, project):
        """Update the API hubs configuration, replacing the previous one.

            hub (str): The hub used for online backend.
            group (str): The group used for online backend.
            project (str): The project used for online backend.
        """
        config_dict = {
            'hub': hub,
            'group': group,
            'project': project
        }

        for key, value in config_dict.items():
            self.__api.config[key] = value
            self.__api_config['config'][key] = value

    def get_api_config(self):
        """Return the program specs."""
        return self.__api_config

    def get_api(self):
        """Returns a function handle to the API."""
        return self.__api

    def save(self, file_name=None, beauty=False):
        """ Save Quantum Program in a Json file.

        Args:
            file_name (str): file name and path.
            beauty (boolean): save the text with indent 4 to make it readable.

        Returns:
            dict: The dictionary with the status and result of the operation

        Raises:
            LookupError: if the file_name is not correct, or writing to the
                file resulted in an error.
        """
        if file_name is None:
            error = {"status": "Error", "result": "Not filename provided"}
            raise LookupError(error['result'])

        if beauty:
            indent = 4
        else:
            indent = 0

        elemements_to_save = self.__quantum_program
        elements_saved = {}

        for circuit in elemements_to_save:
            elements_saved[circuit] = {}
            elements_saved[circuit]["qasm"] = elemements_to_save[circuit].qasm()

        try:
            with open(file_name, 'w') as save_file:
                json.dump(elements_saved, save_file, indent=indent)
            return {'status': 'Done', 'result': elemements_to_save}
        except ValueError:
            error = {'status': 'Error', 'result': 'Some Problem happened to save the file'}
            raise LookupError(error['result'])

    def load(self, file_name=None):
        """ Load Quantum Program Json file into the Quantum Program object.

        Args:
            file_name (str): file name and path.

        Returns:
            dict: The dictionary with the status and result of the operation

        Raises:
            LookupError: if the file_name is not correct, or reading from the
                file resulted in an error.
        """
        if file_name is None:
            error = {"status": "Error", "result": "Not filename provided"}
            raise LookupError(error['result'])

        try:
            with open(file_name, 'r') as load_file:
                elemements_loaded = json.load(load_file)

            for circuit in elemements_loaded:
                circuit_qasm = elemements_loaded[circuit]["qasm"]
                elemements_loaded[circuit] = qasm.Qasm(data=circuit_qasm).parse()
            self.__quantum_program = elemements_loaded

            return {"status": 'Done', 'result': self.__quantum_program}

        except ValueError:
            error = {'status': 'Error', 'result': 'Some Problem happened to load the file'}
            raise LookupError(error['result'])

    def available_backends(self):
        """All the backends that are seen by QISKIT."""
        return self.__ONLINE_BACKENDS + self.__LOCAL_BACKENDS

    def online_backends(self):
        """Get the online backends.

        Queries network API if it exists and gets the backends that are online.

        Returns:
            list(str): List of online backends names if the online api has been set or an empty
                list if it has not been set.

        Raises:
            ConnectionError: if the API call failed.
        """
        if self.get_api():
            try:
                backends = self.__api.available_backends()
            except Exception as ex:
                raise ConnectionError("Couldn't get available backend list: {0}"
                                      .format(ex))
            return [backend['name'] for backend in backends]
        return []

    def online_simulators(self):
        """Gets online simulators via QX API calls.

        Returns:
            list(str): List of online simulator names.

        Raises:
            ConnectionError: if the API call failed.
        """
        online_simulators_list = []
        if self.get_api():
            try:
                backends = self.__api.available_backends()
            except Exception as ex:
                raise ConnectionError("Couldn't get available backend list: {0}"
                                      .format(ex))
            for backend in backends:
                if backend['simulator']:
                    online_simulators_list.append(backend['name'])
        return online_simulators_list

    def online_devices(self):
        """Gets online devices via QX API calls.

        Returns:
            list(str): List of online devices names.

        Raises:
            ConnectionError: if the API call failed.
        """
        devices = []
        if self.get_api():
            try:
                backends = self.__api.available_backends()
            except Exception as ex:
                raise ConnectionError("Couldn't get available backend list: {0}"
                                      .format(ex))
            for backend in backends:
                if not backend['simulator']:
                    devices.append(backend['name'])
        return devices

    def get_backend_status(self, backend):
        """Return the online backend status.

        It uses QX API call or by local backend is the name of the
        local or online simulator or experiment.

        Args:
            backend (str): The backend to check

        Returns:
            dict: {'available': True}

        Raises:
            ConnectionError: if the API call failed.
            ValueError: if the backend is not available.
        """

        if backend in self.__ONLINE_BACKENDS:
            try:
                return self.__api.backend_status(backend)
            except Exception as ex:
                raise ConnectionError("Couldn't get backend status: {0}"
                                      .format(ex))
        elif backend in self.__LOCAL_BACKENDS:
            return {'available': True}
        else:
            raise ValueError('the backend "{0}" is not available'.format(backend))

    def get_backend_configuration(self, backend, list_format=False):
        """Return the configuration of the backend.

        The return is via QX API call.

        Args:
            backend (str):  Name of the backend.
            list_format (bool): Struct used for the configuration coupling
                map: dict (if False) or list (if True).

        Returns:
            dict: The configuration of the named backend.

        Raises:
            ConnectionError: if the API call failed.
            LookupError: if a configuration for the named backend can't be
                found.
        """
        if self.get_api():
            configuration_edit = {}
            try:
                backends = self.__api.available_backends()
            except Exception as ex:
                raise ConnectionError("Couldn't get available backend list: {0}"
                                      .format(ex))
            for configuration in backends:
                if configuration['name'] == backend:
                    for key in configuration:
                        new_key = convert(key)
                        # TODO: removed these from the API code
                        if new_key not in ['id', 'serial_number', 'topology_id',
                                           'status', 'coupling_map']:
                            configuration_edit[new_key] = configuration[key]
                        if new_key == 'coupling_map':
                            if configuration[key] == 'all-to-all':
                                configuration_edit[new_key] = \
                                    configuration[key]
                            else:
                                if not list_format:
                                    cmap = mapper.coupling_list2dict(configuration[key])
                                else:
                                    cmap = configuration[key]
                                configuration_edit[new_key] = cmap
                    return configuration_edit
            raise LookupError('Configuration for %s could not be found.' %
                              backend)
        else:
            return qiskit.backends.get_backend_configuration(backend)

    def get_backend_calibration(self, backend):
        """Return the online backend calibrations.

        The return is via QX API call.

        Args:
            backend (str):  Name of the backend.

        Returns:
            dict: The calibration of the named backend.

        Raises:
            ConnectionError: if the API call failed.
            LookupError: If a configuration for the named backend can't be
                found.
        """
        if backend in self.__ONLINE_BACKENDS:
            try:
                calibrations = self.__api.backend_calibration(backend)
            except Exception as ex:
                raise ConnectionError("Couldn't get backend calibration: {0}"
                                      .format(ex))
            calibrations_edit = {}
            for key, vals in calibrations.items():
                new_key = convert(key)
                calibrations_edit[new_key] = vals
            return calibrations_edit
        elif backend in self.__LOCAL_BACKENDS:
            return {'backend': backend, 'calibrations': None}
        else:
            raise LookupError(
                'backend calibration for "{0}" not found'.format(backend))

    def get_backend_parameters(self, backend):
        """Return the online backend parameters.

        The return is via QX API call.

        Args:
            backend (str):  Name of the backend.

        Returns:
            dict: The configuration of the named backend.

        Raises:
            ConnectionError: if the API call failed.
            LookupError: If a configuration for the named backend can't be
                found.
        """
        if backend in self.__ONLINE_BACKENDS:
            try:
                parameters = self.__api.backend_parameters(backend)
            except Exception as ex:
                raise ConnectionError("Couldn't get backend parameters: {0}"
                                      .format(ex))
            parameters_edit = {}
            for key, vals in parameters.items():
                new_key = convert(key)
                parameters_edit[new_key] = vals
            return parameters_edit
        elif backend in self.__LOCAL_BACKENDS:
            return {'backend': backend, 'parameters': None}
        else:
            raise LookupError(
                'backend parameters for "{0}" not found'.format(backend))

    ###############################################################
    # methods to compile quantum programs into qobj
    ###############################################################

    def compile(self, name_of_circuits, backend="local_qasm_simulator",
                config=None, basis_gates=None, coupling_map=None,
                initial_layout=None, shots=1024, max_credits=10, seed=None,
                qobj_id=None, hpc=None):
        """Compile the circuits into the execution list.

        This builds the internal "to execute" list which is list of quantum
        circuits to run on different backends.

        Args:
            name_of_circuits (list[str]): circuit names to be compiled.
            backend (str): a string representing the backend to compile to.
            config (dict): a dictionary of configurations parameters for the
                compiler.
            basis_gates (str): a comma separated string and are the base gates,
                               which by default are provided by the backend.
            coupling_map (dict): A directed graph of coupling::

                {
                 control(int):
                     [
                         target1(int),
                         target2(int),
                         , ...
                     ],
                     ...
                }

                eg. {0: [2], 1: [2], 3: [2]}

            initial_layout (dict): A mapping of qubit to qubit::

                {
                ("q", strart(int)): ("q", final(int)),
                ...
                }
                eg.
                {
                ("q", 0): ("q", 0),
                ("q", 1): ("q", 1),
                ("q", 2): ("q", 2),
                ("q", 3): ("q", 3)
                }

            shots (int): the number of shots
            max_credits (int): the max credits to use 3, or 5
            seed (int): the initial seed the simulators use
            qobj_id (str): identifier of the qobj.
            hpc (dict): This will setup some parameter for
                ibmqx_hpc_qasm_simulator, using a JSON-like format like::

                    {
                        'multi_shot_optimization': Boolean,
                        'omp_num_threads': Numeric
                    }

                This parameter MUST be used only with
                ibmqx_hpc_qasm_simulator, otherwise the SDK will warn
                the user via logging, and set the value to None.

        Returns:
            dict: the job id and populates the qobj::

            qobj =
                {
                    id: --job id (string),
                    config: -- dictionary of config settings (dict)--,
                        {
                        "max_credits" (online only): -- credits (int) --,
                        "shots": -- number of shots (int) --.
                        "backend": -- backend name (str) --
                        }
                    circuits:
                        [
                            {
                            "name": --circuit name (string)--,
                            "compiled_circuit": --compiled quantum circuit (JSON format)--,
                            "compiled_circuit_qasm": --compiled quantum circuit (QASM format)--,
                            "config": --dictionary of additional config settings (dict)--,
                                {
                                "coupling_map": --adjacency list (dict)--,
                                "basis_gates": --comma separated gate names (string)--,
                                "layout": --layout computed by mapper (dict)--,
                                "seed": (simulator only)--initial seed for the simulator (int)--,
                                }
                            },
                            ...
                        ]
                }

        Raises:
            ValueError: if no names of the circuits have been specified.
            QISKitError: if any of the circuit names cannot be found on the
                Quantum Program.
        """
        # TODO: Jay: currently basis_gates, coupling_map, initial_layout,
        # shots, max_credits and seed are extra inputs but I would like
        # them to go into the config.
        qobj = {}
        if not qobj_id:
            qobj_id = "".join([random.choice(string.ascii_letters+string.digits)
                               for n in range(30)])
        qobj['id'] = qobj_id
        qobj["config"] = {"max_credits": max_credits, 'backend': backend,
                          "shots": shots}

        # TODO This backend needs HPC parameters to be passed in order to work
        if backend == 'ibmqx_hpc_qasm_simulator':
            if hpc is None:
                logger.info('ibmqx_hpc_qasm_simulator backend needs HPC '
                            'parameter. Setting defaults to hpc.multi_shot_optimization '
                            '= true and hpc.omp_num_threads = 16')
                hpc = {'multi_shot_optimization': True, 'omp_num_threads': 16}

            if not all(key in hpc for key in
                       ('multi_shot_optimization', 'omp_num_threads')):
                raise QISKitError('Unknown HPC parameter format!')

            qobj['config']['hpc'] = hpc
        elif hpc is not None:
            logger.info('HPC parameter is only available for '
                        'ibmqx_hpc_qasm_simulator. You are passing an HPC parameter '
                        'but you are not using ibmqx_hpc_qasm_simulator, so we will '
                        'ignore it.')
            hpc = None

        qobj['circuits'] = []
        backend_conf = qiskit.backends.get_backend_configuration(backend)
        if not basis_gates:
            if 'basis_gates' in backend_conf:
                basis_gates = backend_conf['basis_gates']
        elif len(basis_gates.split(',')) < 2:
            # catches deprecated basis specification like 'SU2+CNOT'
            logger.warning('encountered deprecated basis specification: '
                           '"%s" substituting u1,u2,u3,cx,id', str(basis_gates))
            basis_gates = 'u1,u2,u3,cx,id'
        if not coupling_map:
            coupling_map = backend_conf['coupling_map']
        if not name_of_circuits:
            raise ValueError('"name_of_circuits" must be specified')
        if isinstance(name_of_circuits, str):
            name_of_circuits = [name_of_circuits]
        for name in name_of_circuits:
            if name not in self.__quantum_program:
                raise QISKitError('circuit "{0}" not found in program'.format(name))
            # TODO: The circuit object going into this is to have .qasm() method (be careful)
            circuit = self.__quantum_program[name]
            num_qubits = sum((len(qreg) for qreg in circuit.get_qregs().values()))
            # TODO: A better solution is to have options to enable/disable optimizations
            if num_qubits == 1:
                coupling_map = None
            if coupling_map == 'all-to-all':
                coupling_map = None
            dag_circuit, final_layout = openquantumcompiler.compile(
                circuit.qasm(),
                basis_gates=basis_gates,
                coupling_map=coupling_map,
                initial_layout=initial_layout,
                get_layout=True)
            # making the job to be added to qobj
            job = {}
            job["name"] = name
            # config parameters used by the runner
            if config is None:
                config = {}  # default to empty config dict
            job["config"] = copy.deepcopy(config)
            job["config"]["coupling_map"] = mapper.coupling_dict2list(coupling_map)
            # TODO: Jay: make config options optional for different backends
            # Map the layout to a format that can be json encoded
            list_layout = None
            if final_layout:
                list_layout = [[k, v] for k, v in final_layout.items()]
            job["config"]["layout"] = list_layout
            job["config"]["basis_gates"] = basis_gates
            if seed is None:
                job["config"]["seed"] = None
            else:
                job["config"]["seed"] = seed
            # the compiled circuit to be run saved as a dag
            job["compiled_circuit"] = openquantumcompiler.dag2json(dag_circuit,
                                                                   basis_gates=basis_gates)
            # set eval_symbols=True to evaluate each symbolic expression
            # TODO after transition to qobj, we can drop this
            job["compiled_circuit_qasm"] = dag_circuit.qasm(qeflag=True,
                                                            eval_symbols=True)
            # add job to the qobj
            qobj["circuits"].append(job)
        return qobj

    def reconfig(self, qobj, backend=None, config=None, shots=None, max_credits=None, seed=None):
        """Change configuration parameters for a compile qobj. Only parameters which
        don't affect the circuit compilation can change, e.g., the coupling_map
        cannot be changed here!

        Notes:
            If the inputs are left as None then the qobj is not updated

        Args:
            qobj (dict): already compile qobj
            backend (str): see .compile
            config (dict): see .compile
            shots (int): see .compile
            max_credits (int): see .compile
            seed (int): see .compile

        Returns:
            qobj: updated qobj
        """
        if backend is not None:
            qobj['config']['backend'] = backend
        if shots is not None:
            qobj['config']['shots'] = shots
        if max_credits is not None:
            qobj['config']['max_credits'] = max_credits

        for circuits in qobj['circuits']:
            if seed is not None:
                circuits['seed'] = seed
            if config is not None:
                circuits['config'].update(config)

        return qobj

    def get_execution_list(self, qobj, print_func=print):
        """Print the compiled circuits that are ready to run.

        Note:
            This method is intended to be used during interactive sessions, and
            prints directly to stdout instead of using the logger by default. If
            you set print_func with a log function (eg. log.info) it will be used
            instead of the stdout.

        Returns:
            list(str): names of the circuits in `qobj`
        """
        if not qobj:
            print_func("no executions to run")
        execution_list = []

        print_func("id: %s" % qobj['id'])
        print_func("backend: %s" % qobj['config']['backend'])
        print_func("qobj config:")
        for key in qobj['config']:
            if key != 'backend':
                print_func(' ' + key + ': ' + str(qobj['config'][key]))
        for circuit in qobj['circuits']:
            execution_list.append(circuit["name"])
            print_func('  circuit name: ' + circuit["name"])
            print_func('  circuit config:')
            for key in circuit['config']:
                print_func('   ' + key + ': ' + str(circuit['config'][key]))
        return execution_list

    def get_compiled_configuration(self, qobj, name):
        """Get the compiled layout for the named circuit and backend.

        Args:
            name (str):  the circuit name
            qobj (dict): the qobj

        Returns:
            dict: the config of the circuit.

        Raises:
            QISKitError: if the circuit has no configurations
        """
        try:
            for index in range(len(qobj["circuits"])):
                if qobj["circuits"][index]['name'] == name:
                    return qobj["circuits"][index]["config"]
        except KeyError:
            pass
        raise QISKitError('No compiled configurations for circuit "{0}"'.format(name))

    def get_compiled_qasm(self, qobj, name):
        """Return the compiled cricuit in qasm format.

        Args:
            qobj (dict): the qobj
            name (str): name of the quantum circuit

        Returns:
            str: the QASM of the compiled circuit.

        Raises:
            QISKitError: if the circuit has no configurations
        """
        try:
            for index in range(len(qobj["circuits"])):
                if qobj["circuits"][index]['name'] == name:
                    return qobj["circuits"][index]["compiled_circuit_qasm"]
        except KeyError:
            pass
        raise QISKitError('No compiled qasm for circuit "{0}"'.format(name))

    ###############################################################
    # methods to run quantum programs
    ###############################################################

    def run(self, qobj, wait=5, timeout=60):
        """Run a program (a pre-compiled quantum program). This function will
        block until the Job is processed.

        The program to run is extracted from the qobj parameter.

        Args:
            qobj (dict): the dictionary of the quantum object to run.
            wait (int): Time interval to wait between requests for results
            timeout (int): Total time to wait until the execution stops

        Returns:
            Result: A Result (class).
        """
        job_blocker_event = Event()
        job_result = None

        def job_done_callback(results):
            """Callback called when the job is done. It basically
            transforms the results to what the user expects and pass it
            to the main thread
            """
            nonlocal job_result
            job_result = results[0]
            job_blocker_event.set()

        self._run_internal([qobj],
                           wait=wait,
                           timeout=timeout,
                           callback=job_done_callback)

        # Do not set a timeout, as the timeout is being managed by the job
        job_blocker_event.wait()

        return job_result

    def run_batch(self, qobj_list, wait=5, timeout=120):
        """Run various programs (a list of pre-compiled quantum programs). This
        function will block until all programs are processed.

        The programs to run are extracted from qobj elements of the list.

        Args:
            qobj_list (list(dict)): The list of quantum objects to run.
            wait (int): Time interval to wait between requests for results
            timeout (int): Total time to wait until the execution stops

        Returns:
            list(Result): A list of Result (class). The list will contain one Result object
            per qobj in the input list.
        """
        job_blocker_event = Event()
        job_results = []

        def job_done_callback(results):
            """Callback called when the job is done. It basically
            transforms the results to what the user expects and pass it
            to the main thread.
            """
            nonlocal job_results
            job_results = results
            job_blocker_event.set()

        self._run_internal(qobj_list,
                           wait=wait,
                           timeout=timeout,
                           callback=job_done_callback)

        job_blocker_event.wait()
        return job_results

    def run_async(self, qobj, wait=5, timeout=60, callback=None):
        """Run a program (a pre-compiled quantum program) asynchronously. This
        is a non-blocking function, so it will return immediately.

        All input for run comes from qobj.

        Args:
            qobj(dict): the dictionary of the quantum object to
                run or list of qobj.
            wait (int): Time interval to wait between requests for results
            timeout (int): Total time to wait until the execution stops
            callback (fn(result)): A function with signature:
                    fn(result):
                    The result param will be a Result object.
        """

        def job_done_callback(results):
            """Callback called when the job is done. It basically
            transforms the results to what the user expects and pass it
            to the main thread.
            """
            callback(results[0])  # The user is expecting a single Result

        self._run_internal([qobj],
                           wait=wait,
                           timeout=timeout,
                           callback=job_done_callback)

    def run_batch_async(self, qobj_list, wait=5, timeout=120, callback=None):
        """Run various programs (a list of pre-compiled quantum program)
        asynchronously. This is a non-blocking function, so it will return
        immediately.

        All input for run comes from qobj.

        Args:
            qobj_list (list(dict)): The list of quantum objects to run.
            wait (int): Time interval to wait between requests for results
            timeout (int): Total time to wait until the execution stops
            callback (fn(results)): A function with signature:
                    fn(results):
                    The results param will be a list of Result objects, one
                    Result per qobj in the input list.
        """
        self._run_internal(qobj_list,
                           wait=wait,
                           timeout=timeout,
                           callback=callback)

    def _run_internal(self, qobj_list, wait=5, timeout=60, callback=None):
        q_job_list = []
        for qobj in qobj_list:
            q_job = QuantumJob(qobj, preformatted=True, resources={
                'max_credits': qobj['config']['max_credits'], 'wait': wait,
                'timeout': timeout})
            q_job_list.append(q_job)

        job_processor = JobProcessor(q_job_list, max_workers=5,
                                     callback=callback)
        job_processor.submit()

    def execute(self, name_of_circuits, backend="local_qasm_simulator",
                config=None, wait=5, timeout=60, basis_gates=None,
                coupling_map=None, initial_layout=None, shots=1024,
                max_credits=3, seed=None, hpc=None):

        """Execute, compile, and run an array of quantum circuits).

        This builds the internal "to execute" list which is list of quantum
        circuits to run on different backends.

        Args:
            name_of_circuits (list[str]): circuit names to be compiled.
            backend (str): a string representing the backend to compile to
            config (dict): a dictionary of configurations parameters for the
                compiler
            wait (int): Time interval to wait between requests for results
            timeout (int): Total time to wait until the execution stops
            basis_gates (str): a comma separated string and are the base gates,
                               which by default are: u1,u2,u3,cx,id
            coupling_map (dict): A directed graph of coupling::

                                {
                                control(int):
                                    [
                                        target1(int),
                                        target2(int),
                                        , ...
                                    ],
                                    ...
                                }
                                eg. {0: [2], 1: [2], 3: [2]}
            initial_layout (dict): A mapping of qubit to qubit
                                  {
                                  ("q", start(int)): ("q", final(int)),
                                  ...
                                  }
                                  eg.
                                  {
                                  ("q", 0): ("q", 0),
                                  ("q", 1): ("q", 1),
                                  ("q", 2): ("q", 2),
                                  ("q", 3): ("q", 3)
                                  }
            shots (int): the number of shots
            max_credits (int): the max credits to use 3, or 5
            seed (int): the initial seed the simulators use
            hpc (dict): This will setup some parameter for
                        ibmqx_hpc_qasm_simulator, using a JSON-like format like::

                            {
                                'multi_shot_optimization': Boolean,
                                'omp_num_threads': Numeric
                            }

                        This parameter MUST be used only with
                        ibmqx_hpc_qasm_simulator, otherwise the SDK will warn
                        the user via logging, and set the value to None.

        Returns:
            Result: status done and populates the internal __quantum_program with the
            data
        """
        # TODO: Jay: currently basis_gates, coupling_map, intial_layout, shots,
        # max_credits, and seed are extra inputs but I would like them to go
        # into the config
        qobj = self.compile(name_of_circuits, backend=backend, config=config,
                            basis_gates=basis_gates,
                            coupling_map=coupling_map, initial_layout=initial_layout,
                            shots=shots, max_credits=max_credits, seed=seed,
                            hpc=hpc)
        result = self.run(qobj, wait=wait, timeout=timeout)
        return result
