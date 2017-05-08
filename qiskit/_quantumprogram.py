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

Authors: Andrew Cross, Jay M. Gambetta, Ismael Faro
"""
# pylint: disable=line-too-long

import time
import json
from collections import Counter

# use the external IBMQuantumExperience Library
from IBMQuantumExperience import IBMQuantumExperience
# stable Modules
from . import QuantumRegister
from . import ClassicalRegister
from . import QuantumCircuit
# Beta Modules
from . import unroll    
from . import qasm
# from .qasm import QasmException

from .extensions.standard import barrier, h, cx, u3, x, z

class QuantumProgram(object):
    """ Quantum Program Class

     Class internal properties """
    __specs = {}
    __quantum_registers = {}
    __classical_registers = {}
    __circuits = {}
    __API = {}
    __API_config = {}

    def __init__(self, specs=None, name="", circuit=None, scope=None):
        # with open('config.json') as data_file:
        #     config = json.load(data_file)

        # self.__API_config = config["API"]
        # self.__QASM = qasm.Qasm()
        self.__circuits = {}
        self.__quantum_registers = {}
        self.__classical_registers = {}
        self.__scope = scope
        self.__name = name
        if specs:
            self.__init_specs(specs)
        if circuit:
            self.__circuits[circuit["name"]] = (circuit)

    def quantum_elements(self, specs=None):
        """Return the basic elements, Circuit, Quantum Registers, Classical Registers"""
        if not specs:
            specs = self.get_specs()

        return self.__circuits[list(self.__circuits)[0]], \
            self.__quantum_registers[list(self.__quantum_registers)[0]], \
            self.__classical_registers[list(self.__classical_registers)[0]]

    def quantum_registers(self, name):
        """Return a specific Quantum Registers"""
        return self.__quantum_registers[name]

    def classical_registers(self, name):
        """Return a specific Classical Registers"""
        return self.__classical_registers[name]

    def circuit(self, name):
        """Return a specific Circuit"""
        return self.__circuits[name]

    def get_specs(self):
        """Return the program specs"""
        return self.__specs

    def api_config(self):
        """Return the program specs"""
        return self.__API.req.credential.config

    def _setup_api(self, token, url):
        try:
            self.__API = IBMQuantumExperience.IBMQuantumExperience(token, {"url":url})
            return True
        except:
            print('Exception connect to servers')
            return False

    def set_api(self, token=None, url=None):
        """Set the API conf"""
        if not token:
            token = self.__API_config["token"]
        else:
            self.__API_config["token"] = token
        if not url:
            url = self.__API_config["url"]
        else:
            self.__API_config["url"] = {"url":url}
        return self._setup_api(token, url)


    def set_api_token(self, token):
        """ Set the API Token """
        self.set_api(token=token)

    def set_api_url(self, url):
        """ Set the API url """
        self.set_api(url=url)

    def get_job_list_status(self, jobids):
        """Given a list of job ids, return a list of job status.
        jobids is a list of id strings.
        api is an IBMQuantumExperience object.
        """
        status_list = []
        for i in jobids:
            status_list.append(self.__API.get_job(i)['status'])
        return status_list

    def unroller_code(self, circuit, basis_gates=None):
        """ Unroller the code
        circuits are circuits to unroll
        asis_gates are the base gates by default are: u1,u2,u3,cx
        """
        if not basis_gates:
            basis_gates = "u1,u2,u3,cx"  # QE target basis

        unrolled_circuit = unroll.Unroller(qasm.Qasm(data=circuit.qasm()).parse(),
                                           unroll.CircuitBackend(basis_gates.split(",")))
        unrolled_circuit.execute()

        circuit_unrolled = unrolled_circuit.backend.circuit  # circuit DAG
        qasm_source = circuit_unrolled.qasm(qeflag=True)
        return qasm_source

    def run_circuits(self, circuits, device, shots, max_credits=3, basis_gates=None):
        """Run a circuit.
        circuit is a circuit name
        api the api for the device
        device is a string for real or simulator
        shots is the number of shots
        max_credits is the credits of the experiments.
        basis_gates are the base gates by default are: u1,u2,u3,cx
        """
        jobs = []
        for circuit in circuits:
            jobs.append({'qasm': self.unroller_code(circuit, basis_gates)})
        output = self.__API.run_job(jobs, device, shots, max_credits)
        return output

    def run_circuit(self, circuit, device, shots, max_credits=3, basis_gates=None):
        """Run a circuit.
        circuit is a circuit name
        api the api for the device
        device is a string for real or simulator
        shots is the number of shots
        max_credits is the credits of the experiments.
        basis_gates are the base gates by default are: u1,u2,u3,cx
        """
        if not self.__API:
            return {"status":"Error", "result":"Not API setup"}
        if isinstance(circuit, str):
            circuit = self.__circuits[circuit]

        qasm_source = self.unroller_code(circuit)
        output = self.__API.run_experiment(qasm_source, device, shots, max_credits)
        return output

    def run_program(self, device, shots, max_credits=3, basis_gates=None):
        """Run a program (array of quantum circuits).
        program is a list of quantum_circuits
        api the api for the device
        device is a string for real or simulator
        shots is the number of shots
        max_credits is the credits of the experiments.
        basis_gates are the base gates by default are: u1,u2,u3,cx
        """
        output = self.run_circuits(self.__circuits.values(), device, shots, max_credits=3, basis_gates=None)
        return output

    def program_to_text(self, circuits=None):
        """Print a program (array of quantum circuits).

        program is a list of quantum circuits, if it's emty use the internal circuits
        """
        if not circuits:
            circuits = self.__circuits.values()

        jobs = ""
        for circuit in circuits:
            jobs = jobs + self.unroller_code(circuit) + "\n\n"
        return jobs

    def wait_for_jobs(self, jobids, wait=5, timeout=60):
        """Wait until all status results are 'COMPLETED'.
        jobids is a list of id strings.
        api is an IBMQuantumExperience object.
        wait is the time to wait between requests, in seconds
        timeout is how long we wait before failing, in seconds
        Returns an list of results that correspond to the jobids.
        """
        status = dict(Counter(self.get_job_list_status(jobids)))
        t = 0
        timeout_over = False
        print("status = %s (%d seconds)" % (status, t))
        while 'COMPLETED' not in status or status['COMPLETED'] < len(jobids):
            if t == timeout:
                timeout_over = True
                break
            time.sleep(wait)
            t += wait
            status = dict(Counter(self.get_job_list_status(jobids)))
            print("status = %s (%d seconds)" % (status, t))
        # Get the results
        results = []

        if timeout_over:
            return {"status":"Error", "result":"Time Out"}

        for i in jobids:
            results.append(self.__API.get_job(i))
        return results

    def combine_jobs(self, jobids, wait=5, timeout=60):
        """Like wait_for_jobs but with a different return format.
        jobids is a list of id strings.
        api is an IBMQuantumExperience object.
        wait is the time to wait between requests, in seconds
        timeout is how long we wait before failing, in seconds
        Returns a list of dict outcomes of the flattened in the order
        jobids so it works with _getData_. """

        results = list(map(lambda x: x['qasms'],
                           self.wait_for_jobs(jobids, wait, timeout)))
        flattened = []
        for sublist in results:
            for val in sublist:
                flattened.append(val)
        # Are there other parts from the dictionary that we want to add,
        # such as shots?
        return {'qasms': flattened}


    def average_data(self, data, observable):
        """Compute the mean value of an observable.
        Takes in the data counts(i) and a corresponding observable in dict
        form and calculates sum_i value(i) P(i) where value(i) is the value of
        the observable for the i state.
        """
        temp = 0
        tot = sum(data.values())
        for key in data:
            if key in observable:
                temp += data[key]*observable[key]/tot
        return temp

    def __init_specs(self, specs):
        """Populate the Quantum Program Object with initial Specs"""
        self.__specs = specs
        quantumr = []
        classicalr = []
        if "api" in specs:
            if  specs["api"]["token"]:
                self.__API_config["token"] = specs["api"]["token"]
            if  specs["api"]["url"]:
                self.__API_config["url"] = specs["api"]["url"]

        if "circuits" in specs:
            for circuit in specs["circuits"]:
                quantumr = self.create_quantum_registers_group(circuit["quantum_registers"])
                classicalr = self.create_classical_registers_group(circuit["classical_registers"])
                self.create_circuit(name=circuit["name"],
                                    qregisters=quantumr,
                                    cregisters=classicalr)
        else:
            if "quantum_registers" in specs:
                print("quantum_registers created")
                quantumr = specs["quantum_registers"]
                self.create_quantum_registers(quantumr["name"], quantumr["size"])
            if "classical_registers" in specs:
                print("quantum_registers created")
                classicalr = specs["classical_registers"]
                self.create_classical_registers(classicalr["name"], classicalr["size"])
            if quantumr and classicalr:
                self.create_circuit(name=specs["name"],
                                    qregisters=quantumr["name"],
                                    cregisters=classicalr["name"])

    def create_circuit(self, name, qregisters, cregisters):
        """Create a new Quantum Circuit into the Quantum Program
        name is a string, the name of the circuit
        qregisters is a Array of Quantum Registers, can be String, by name or the object reference
        cregisters is a Array of Classical Registers, can be String, by name or the object reference
        """
        self.__circuits[name] = QuantumCircuit()
        for register in qregisters:
            if isinstance(register, str):
                self.__circuits[name].add(self.__quantum_registers[register])
            else:
                self.__circuits[name].add(register)
        for register in cregisters:
            if isinstance(register, str):
                self.__circuits[name].add(self.__classical_registers[register])
            else:
                self.__circuits[name].add(register)
        return self.__circuits[name]

    def create_quantum_registers(self, name, size):
        """Create a new set of Quantum Registers"""
        self.__quantum_registers[name] = QuantumRegister(name, size)
        print(">> quantum_registers created:", name, size)
        return self.__quantum_registers[name]

    def create_quantum_registers_name(self, name, size):
        """Create a new set of Quantum Registers"""
        self.__quantum_registers[name] = QuantumRegister(name, size)
        print(">> quantum_registers created:", name, size)
        return self.__quantum_registers[name]

    def create_quantum_registers_group(self, registers_array):
        """Create a new set of Quantum Registers based in a array of that"""
        new_registers = []
        for register in registers_array:
            register = self.create_quantum_registers(register["name"], register["size"])
            new_registers.append(register)
        print(new_registers)
        return new_registers

    def create_classical_registers_group(self, registers_array):
        """Create a new set of Classical Registers based in a array of that"""
        new_registers = []
        for register in registers_array:
            new_registers.append(self.create_classical_registers(register["name"], register["size"]))
        return new_registers

    def create_classical_registers(self, name, size):
        """Create a new set of Classical Registers"""
        self.__classical_registers[name] = ClassicalRegister(name, size)
        print(">> classical_registers created:", name, size)
        return self.__classical_registers[name]
        