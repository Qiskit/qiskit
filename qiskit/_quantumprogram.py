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
from collections import Counter
# use the external IBMQuantumExperience Library
from IBMQuantumExperience.IBMQuantumExperience import IBMQuantumExperience
from . import basicplotter

# stable Modules
from . import QuantumRegister
from . import ClassicalRegister
from . import QuantumCircuit

# Beta Modules
from . import unroll
from . import qasm
from . import mapper


class QuantumProgram(object):
    """ Quantum Program Class

     Class internal properties """
    __specs = {}
    __quantum_registers = {}
    __classical_registers = {}
    __circuits = {}
    __api = {}
    __api_config = {}
    __qasm_compile = {
        'backend': {'name': 'simulator'},
        'max_credits': 3,
        'circuits': [],
        'compiled_circuits': [],
        'shots': 1024
    }

    """
        objects examples:

        qasm_compile=
            {
                'backend': {'name': 'qx5qv2'},
                'max_credits': 3,
                'id': 'id0000',
                'circuits':  [
                    {'qasm': 'OPENQASM text version of circuit 1 from user input'},
                    {'qasm': 'OPENQASM text version of circuit 2 from user input'}
                    ],
                'compiled_circuits': [
                    {’qasm’: ’Compiled QASM text version of circuit 1 to run on device’,
                    'exucution_id': 'id000',
                    'result': {
                        'data':{
                            'counts': {’00000’: XXXX, ’00001’: XXXXX},
                            'time'  : xx.xxxxxxxx},
                            ’date’  : ’2017−05−09Txx:xx:xx.xxxZ’
                            },
                        ’status’: ’DONE’}
                    },
                    {’qasm’: ’text version of circuit 2 to run on device’, },
                ],
                'shots': 1024,
            }

    """

    def __init__(self, specs=None, name="", circuit=None, scope=None):
        self.__circuits = {}
        self.__quantum_registers = {}
        self.__classical_registers = {}
        self.__scope = scope
        self.__name = name
        # self.__api = {}

        self.mapper = mapper

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
        return self.__api.req.credential.config

    def _setup_api(self, token, url):
        try:
            self.__api = IBMQuantumExperience(token, {"url": url})
            return True
        except BaseException:
            print('---- Error: Exception connect to servers ----')

            return False

    def set_api(self, token=None, url=None):
        """Set the API conf"""
        if not token:
            token = self.__api_config["token"]
        else:
            self.__api_config["token"] = token
        if not url:
            url = self.__api_config["url"]
        else:
            self.__api_config["url"] = {"url": url}
        api = self._setup_api(token, url)
        return api

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
            status_list.append(self.__api.get_job(i)['status'])
        return status_list

    def load_qasm(self, qasm_file, basis_gates=None):
        """ Load qasm file
        qasm_file qasm file name
        """
        if not basis_gates:
            basis_gates = "u1,u2,u3,cx,id"  # QE target basis

        try:
            qasm_file = open(qasm_file, 'r')
            qasm_source = qasm_file.read()
            qasm_file.close()
            circuit = unroll.Unroller(qasm.Qasm(data=qasm_source).parse(),
                                      unroll.CircuitBackend(basis_gates.split(",")))
            # print(circuit.qasm())
            self.__circuits[qasm_file] = circuit
            return circuit
        except BaseException:
            print('---- Error: Load qasm file = ', qasm_file)

    def unroller_code(self, circuit, basis_gates=None):
        """ Unroller the code
        circuits are circuits to unroll
        asis_gates are the base gates by default are: u1,u2,u3,cx,id
        """
        if not basis_gates:
            basis_gates = "u1,u2,u3,cx,id"  # QE target basis

        unrolled_circuit = unroll.Unroller(qasm.Qasm(data=circuit.qasm()).parse(),
                                           unroll.CircuitBackend(basis_gates.split(",")))
        unrolled_circuit.execute()

        circuit_unrolled = unrolled_circuit.backend.circuit  # circuit DAG
        qasm_source = circuit_unrolled.qasm(qeflag=True)
        return qasm_source, circuit_unrolled

    def circuits_qasm(self, circuits):
        qasm_circuits = []
        for circuit in circuits:
            qasm_circuits.append({'qasm': circuit.qasm()})
        return qasm_circuits

    def compile(self, device, coupling_map=None, shots=1024, max_credits=3):
        """ Compile unrole the code
        circuits are circuits to unroll
        basis_gates are the base gates by default are: u1,u2,u3,cx,id
        """
        self.__qasm_compile = {
            'backend': {'name': device},
            'max_credits': max_credits,
            'compiled_circuits': self.compile_circuits(self.__circuits.values(), coupling_map=coupling_map)[0],
            'shots': shots
        }

        return self.__qasm_compile

    def compile_circuits(self, circuits, coupling_map=None):
        """ Compile unrole the code
        circuits are circuits to unroll
        basis_gates are the base gates by default are: u1,u2,u3,cx,id
        """
        qasm_circuits = []
        unrolled_circuits = []
        for circuit in circuits:
            qasm_source, circuit_unrolled = self.unroller_code(circuit)
            if coupling_map:
                print("pre-mapping properties: %s"
                      % circuit_unrolled.property_summary())
                # Insert swap gates
                coupling = self.mapper.Coupling(coupling_map)
                circuit_unrolled, final_layout = self.mapper.swap_mapper(
                    circuit_unrolled, coupling)
                print("layout: %s" % final_layout)
                # Expand swaps
                qasm_source, circuit_unrolled = self.unroller_code(
                    circuit_unrolled)
                # Change cx directions
                circuit_unrolled = mapper.direction_mapper(circuit_unrolled,
                                                           coupling)
                # Simplify cx gates
                mapper.cx_cancellation(circuit_unrolled)
                # Simplify single qubit gates
                circuit_unrolled = mapper.optimize_1q_gates(circuit_unrolled)
                qasm_source = circuit_unrolled.qasm(qeflag=True)
                print("post-mapping properties: %s"
                      % circuit_unrolled.property_summary())
            qasm_circuits.append({'qasm': qasm_source})
            unrolled_circuits.append({'circuit_unrolled': circuit_unrolled})
        return qasm_circuits, unrolled_circuits

    def run(self, wait=5, timeout=60):
        """Run a program (array of quantum circuits).
        program is a list of quantum_circuits
        api the api for the device
        device is a string for real or simulator
        shots is the number of shots
        max_credits is the credits of the experiments.
        basis_gates are the base gates by default are: u1,u2,u3,cx,id
        """
        output = self.__api.run_job(self.__qasm_compile['compiled_circuits'],
                                    self.__qasm_compile['backend']['name'],
                                    self.__qasm_compile['shots'],
                                    self.__qasm_compile['max_credits'])
        if 'error' in output:
            return {"status": "Error", "result": output['error']}

        job_result = self.wait_for_job(
            output['id'], wait=wait, timeout=timeout)

        if job_result['status'] == 'Error':
            return job_result

        self.__qasm_compile['compiled_circuits'] = job_result['qasms']
        self.__qasm_compile['used_credits'] = job_result['usedCredits']
        self.__qasm_compile['status'] = job_result['status']

        return self.__qasm_compile

    def run_circuits(self, circuits, device, shots,
                     max_credits=3, basis_gates=None, wait=5, timeout=60):
        """Run a circuit.
        circuit is a circuit name
        api the api for the device
        device is a string for real or simulator
        shots is the number of shots
        max_credits is the credits of the experiments.
        basis_gates are the base gates by default are: u1,u2,u3,cx,id
        """
        jobs = []
        for circuit in circuits:
            qasm_source, circuit = self.unroller_code(circuit, basis_gates)
            jobs.append({'qasm': qasm_source})
        output = self.__api.run_job(jobs, device, shots, max_credits)
        if 'error' in output:
            return {"status": "Error", "result": output['error']}

        job_result = self.wait_for_job(
            output['id'], wait=wait, timeout=timeout)

        return job_result

    def run_circuit(self, circuit, device, shots, max_credits=3):
        """Run a circuit.
        circuit is a circuit name
        api the api for the device
        device is a string for real or simulator
        shots is the number of shots
        max_credits is the credits of the experiments.
        basis_gates are the base gates by default are: u1,u2,u3,cx,id
        """
        if not self.__api:
            return {"status": "Error", "result": "Not API setup"}
        if isinstance(circuit, str):
            circuit = self.__circuits[circuit]

        qasm_source, circuit = self.unroller_code(circuit)
        output = self.__api.run_experiment(
            qasm_source, device, shots, max_credits)
        return output

    def run_program(self, device, shots, max_credits=3, basis_gates=None):
        """Run a program (array of quantum circuits).
        program is a list of quantum_circuits
        api the api for the device
        device is a string for real or simulator
        shots is the number of shots
        max_credits is the credits of the experiments.
        basis_gates are the base gates by default are: u1,u2,u3,cx,id
        """
        output = self.run_circuits(self.__circuits.values(
        ), device, shots, max_credits=max_credits, basis_gates=basis_gates)
        return output

    def execute(self, device='simulator', coupling_map=None,
                shots=1024, max_credits=3):
        """Execute compile and run a program (array of quantum circuits).
        program is a list of quantum_circuits
        api the api for the device
        device is a string for real or simulator
        shots is the number of shots
        max_credits is the credits of the experiments.
        basis_gates are the base gates by default are: u1,u2,u3,cx,id
        """
        self.compile(device, coupling_map, shots, max_credits)
        output = self.run()

        return output

    def execute_circuits(self, circuits, device, shots,
                         max_credits=3, basis_gates=None):
        """Execute compile and run a program (array of quantum circuits).
        program is a list of quantum_circuits
        api the api for the device
        device is a string for real or simulator
        shots is the number of shots
        max_credits is the credits of the experiments.
        basis_gates are the base gates by default are: u1,u2,u3,cx,id
        """
        qasm_source, circuit_unrolled = self.compile()
        output = self.run_circuits(
            circuit_unrolled, device, shots, max_credits=max_credits, basis_gates=basis_gates)
        return output

    def program_to_text(self, circuits=None):
        """Print a program (array of quantum circuits).

        program is a list of quantum circuits, if it's emty use the internal circuits
        """
        if not circuits:
            circuits = self.__circuits.values()
        # TODO: Store QASM per circuit
        jobs = ""
        for circuit in circuits:
            qasm_source, circuit = self.unroller_code(circuit)
            jobs = jobs + qasm_source + "\n\n"
        return jobs[:-3]

    def wait_for_job(self, jobid, wait=5, timeout=60):
        """Wait until all status results are 'COMPLETED'.
        jobids is a list of id strings.
        api is an IBMQuantumExperience object.
        wait is the time to wait between requests, in seconds
        timeout is how long we wait before failing, in seconds
        Returns an list of results that correspond to the jobids.
        """
        timer = 0
        timeout_over = False
        job = self.__api.get_job(jobid)
        while job['status'] == 'RUNNING':
            if timer == timeout:
                timeout_over = True
                break
            time.sleep(wait)
            timer += wait
            print("status = %s (%d seconds)" % (job['status'], timer))
            job = self.__api.get_job(jobid)
            if job['status'] == 'ERROR_CREATING_JOB' or job['status'] == 'ERROR_RUNNING_JOB':
                return {"status": "Error", "result": job['status']}
        # Get the results

        if timeout_over:
            return {"status": "Error", "result": "Time Out"}
        return job

    def wait_for_jobs(self, jobids, wait=5, timeout=60):
        """Wait until all status results are 'COMPLETED'.
        jobids is a list of id strings.
        api is an IBMQuantumExperience object.
        wait is the time to wait between requests, in seconds
        timeout is how long we wait before failing, in seconds
        Returns an list of results that correspond to the jobids.
        """
        status = dict(Counter(self.get_job_list_status(jobids)))
        timer = 0
        timeout_over = False
        print("status = %s (%d seconds)" % (status, timer))
        while 'COMPLETED' not in status or status['COMPLETED'] < len(jobids):
            if timer == timeout:
                timeout_over = True
                break
            time.sleep(wait)
            timer += wait
            status = dict(Counter(self.get_job_list_status(jobids)))
            print("status = %s (%d seconds)" % (status, timer))
        # Get the results
        results = []

        if timeout_over:
            return {"status": "Error", "result": "Time Out"}

        for i in jobids:
            results.append(self.__api.get_job(i))
        return results

    def flat_results(self, results):
        """ Flat the results
        results array of results
        """
        flattened = []
        for sublist in results:
            for val in sublist:
                flattened.append(val)
        return {'qasms': flattened}

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
        return self.flat_results(results)

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
                temp += data[key] * observable[key] / tot
        return temp

    def __init_specs(self, specs):
        """Populate the Quantum Program Object with initial Specs"""
        self.__specs = specs
        quantumr = []
        classicalr = []
        if "api" in specs:
            if specs["api"]["token"]:
                self.__api_config["token"] = specs["api"]["token"]
            if specs["api"]["url"]:
                self.__api_config["url"] = specs["api"]["url"]

        if "circuits" in specs:
            for circuit in specs["circuits"]:
                quantumr = self.create_quantum_registers_group(
                    circuit["quantum_registers"])
                classicalr = self.create_classical_reg_group(
                    circuit["classical_registers"])
                self.create_circuit(name=circuit["name"],
                                    qregisters=quantumr,
                                    cregisters=classicalr)
        else:
            if "quantum_registers" in specs:
                print(">> quantum_registers created")
                quantumr = specs["quantum_registers"]
                self.create_quantum_registers(
                    quantumr["name"], quantumr["size"])
            if "classical_registers" in specs:
                print(">> quantum_registers created")
                classicalr = specs["classical_registers"]
                self.create_classical_registers(
                    classicalr["name"], classicalr["size"])
            if quantumr and classicalr:
                self.create_circuit(name=specs["name"],
                                    qregisters=quantumr["name"],
                                    cregisters=classicalr["name"])

    def create_circuit(self, name, qregisters=None, cregisters=None):
        """Create a new Quantum Circuit into the Quantum Program
        name is a string, the name of the circuit
        qregisters is a Array of Quantum Registers, can be String, by name or the object reference
        cregisters is a Array of Classical Registers, can be String, by name or the object reference
        """
        if not qregisters:
            qregisters = []
        if not cregisters:
            cregisters = []

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
            register = self.create_quantum_registers(
                register["name"], register["size"])
            new_registers.append(register)
        return new_registers

    def create_classical_reg_group(self, registers_array):
        """Create a new set of Classical Registers based in a array of that"""
        new_registers = []
        for register in registers_array:
            new_registers.append(self.create_classical_registers(
                register["name"], register["size"]))
        return new_registers

    def create_classical_registers(self, name, size):
        """Create a new set of Classical Registers"""
        self.__classical_registers[name] = ClassicalRegister(name, size)
        print(">> classical_registers created:", name, size)
        return self.__classical_registers[name]

    def plotter(self, method="histogram", circuit_number=0):
        """Plot the results
        method: histogram/qsphere
        circuit: Print one circuit
        """
        if self.__qasm_compile == {}:
            print("---- Error: Not data to plot ----")
            return None
        if not self.__qasm_compile["status"] == "DONE":
            print("---- Errorr: No results, status = ",
                  self.__qasm_compile["status"])
            return None

        data = self.__qasm_compile['compiled_circuits'][circuit_number]['result']['data']['count']
        print(data)
        if method == "histogram":
            basicplotter.plot_histogram(data, circuit_number)
        else:
            basicplotter.plot_qsphere(data, circuit_number)
