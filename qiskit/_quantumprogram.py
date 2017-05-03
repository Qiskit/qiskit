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

from . import unroll
from . import qasm
from . import QuantumRegister
from . import ClassicalRegister
from . import QuantumCircuit
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
    # __QASM = {}

    def __init__(self, specs, name="", circuit=None, scope=None):
        with open('config.json') as data_file:
            config = json.load(data_file)

        self.__API_config = config["API"]
        # self.__QASM = qasm.Qasm()
        self.__scope = scope
        self.__name = name
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
        except:
            print('Exception connect to servers')

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
        self._setup_api(token, url)

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

    def run_circuit(self, circuit ,device, shots, max_credits=3, basis=None):
        """Run a circuit.
        circuit is a circuit name
        api the api for the device
        device is a string for real or simulator
        shots is the number of shots
        max_credits is the credits of the experiments.
        """
        if not basis:
            basis = "u1,u2,u3,cx"  # QE target basis

        unrolled_circuit = unroll.Unroller(qasm.Qasm(data=self.__circuits[circuit].qasm()).parse(),
                                           unroll.CircuitBackend(basis.split(",")))
        # print(unrolled_circuit.)
        unrolled_circuit.execute()

        C = unrolled_circuit.backend.circuit  # circuit DAG
        qasm_source = C.qasm(qeflag=True)
        print(qasm_source)
        output = self.__API.run_experiment(qasm_source, device, shots, max_credits)
        return output

    def run_program(self, device, shots, max_credits=3):
        """Run a program (array of quantum circuits).
        program is a list of quantum_circuits
        api the api for the device
        device is a string for real or simulator
        shots is the number of shots
        max_credits is the credits of the experiments.
        """
        jobs = []
        for circuit in self.__circuits:
            basis = "u1,u2,u3,cx"  # QE target basis
            unrolled_circuit = unroll.Unroller(qasm.Qasm(data=circuit.qasm()).parse(),
                                               unroll.CircuitBackend(basis.split(",")))
            # unrolled_circuit = unroll.Unroller(Qasm(data=circuit.qasm()).parse(), unroll.CircuitBackend(
            #                                    basis.split(",")))
            unrolled_circuit.execute()
            C = unrolled_circuit.backend.circuit # circuit DAG

            jobs.append({'qasm': C.qasm(qeflag=True)})
        out = self.__API.run_job(jobs, device, shots, max_credits)
        return out

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
        print("status = %s (%d seconds)" % (status, t))
        while 'COMPLETED' not in status or status['COMPLETED'] < len(jobids):
            if t == timeout:
                break
            time.sleep(wait)
            t += wait
            status = dict(Counter(self.get_job_list_status(jobids)))
            print("status = %s (%d seconds)" % (status, t))
        # Get the results
        results = []
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
        quantumr = {}
        classicalr = {}
        if "api" in specs:
            if  specs["api"]["token"]:
                self.__API_config["token"] = specs["api"]["token"]
            if  specs["api"]["url"]:
                self.__API_config["url"] = specs["api"]["url"]

        if "circuits" in specs:
            for circuit in specs["circuits"]:
                quantumr = circuit["quantum_registers"]
                self.create_quantum_registers(quantumr["name"], quantumr["size"])
                classicalr = circuit["classical_registers"]
                self.create_classical_registers(classicalr["name"], classicalr["size"])
                self.create_circuit(name=circuit["name"],
                                    qregisters=quantumr["name"],
                                    cregisters=classicalr["name"])
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
        """Create a new Quantum Circuit into the Quantum Program"""
        print(name)
        self.__circuits[name] = "demo"
        self.__circuits[name] = QuantumCircuit(self.__quantum_registers[qregisters],
                                               self.__classical_registers[cregisters])
        print(">> circuit created")
        return self.__circuits[name]


    def create_quantum_registers(self, name, size):
        """Create a new set of Quantum Registers"""
        self.__quantum_registers[name] = QuantumRegister(name, size)
        print(">> quantum_registers created:", name, size)
        return self.__quantum_registers[name]

    def create_classical_registers(self, name, size):
        """Create a new set of Classical Registers"""
        self.__classical_registers[name] = ClassicalRegister(name, size)
        print(">> classical_registers created:", name, size)
        return self.__classical_registers[name]