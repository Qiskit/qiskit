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
import random
from collections import Counter
# use the external IBMQuantumExperience Library
from IBMQuantumExperience.IBMQuantumExperience import IBMQuantumExperience
from . import basicplotter

# stable Modules
from . import QuantumRegister
from . import ClassicalRegister
from . import QuantumCircuit
from . import QISKitException

# Beta Modules
from . import unroll
from . import qasm
from . import mapper

from .unroll import SimulatorBackend
from .simulators._unitarysimulator import UnitarySimulator
from .simulators._qasmsimulator import QasmSimulator

import sys
sys.path.append("..")
from qiskit.extensions.standard import x, h, cx, s, ry, barrier


class QuantumProgram(object):
    """ Quantum Program Class

     Class internal properties """
    __online_devices = ["IBMQX5qv2", "ibmqx2", "ibmqx3", "ibmqx_qasm_simulator", "simulator"]
    __local_devices = ["local_unitary_simulator", "local_qasm_simulator"]

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
    __qprogram = {}

    """
        objects examples:

        qasm_compile=
            {
                'backend': {'name': 'ibmqx2'},
                'max_credits': 3,
                'id': 'id0000',
                'circuits':  [
                    {'qasm': 'OPENQASM text version of circuit 1 from user input'},
                    {'qasm': 'OPENQASM text version of circuit 2 from user input'}
                    ],
                'compiled_circuits': [
                    {
                    'name': #TODO: use the name to update the compile
                    ’qasm’: ’Compiled QASM to run on backend, #TODO: convert to object
                    'execution_id': 'id000',
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
#       'circuits': "name_circuit" {
#                     {
#                     'name': #TODO: use the name to update the compile
#                     'QASM_source': ’Compiled QASM to run on backend, #TODO: convert to object
#                     'QASM_compiled':
#                     'execution_id': 'id000',
#                     'result': {
#                         'data':{
#                             'counts': {’00000’: XXXX, ’00001’: XXXXX},
#                             'time'  : xx.xxxxxxxx},
#                             ’date’  : ’2017−05−09Txx:xx:xx.xxxZ’
#                             },
#                         ’status’: ’DONE’}
#                     }}

#
#    backend =  {device, shots, max_credits, }
#       'circuit_to_execute': [{a,device1,shots,max_credit},{b,device1},{a,device2},{b,device2},.......],
#       'circuits': { "a": {
#                     'QASM': ’Compiled QASM to run on backend, #TODO: convert to object
#                     'execution: {'local_simulatior': { QASM_compile, data, shots, status}
#                       }
#                     .....},
#                     "b": {
#
#                     'QASM': ’Compiled QASM to run on backend, #TODO: convert to object
#                     'QASM_compiled: None
#                     .....},
#                     "name": {
#                     'QASM': ’New'
#                     'QASM_compiled: None
#                     .....},
# }

# c = qp.add('name',a, b)

# qp.compile(['a','b'],backend, ....)
# qp.compile(['a','b'],backend, ....)
# qp.compile(['a','b'],device2, ....)
# qp.run(....)

    def __init__(self, specs=None, name=""):
        self.__circuits = {"circuits":{}}
        self.__quantum_registers = {}
        self.__classical_registers = {}

        self.__name = name
        self.__qprogram = {}
        # self.__api = {}

        self.mapper = mapper

        if specs:
            self.__init_specs(specs)

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

    def load_qasm(self, name="", qasm_file=None, basis_gates=None):
        """ Load qasm file
        qasm_file qasm file name
        """
        if not qasm_file:
            print('"Not filename provided')
            return {"status": "Error", "result": "Not filename provided"}
        if not basis_gates:
            basis_gates = "u1,u2,u3,cx,id"  # QE target basis

        if name == "":
            name = qasm_file

        circuit = qasm.Qasm(filename=qasm_file).parse()


        self.__circuits['circuits'][name] = {"name":name, "object": circuit, "QASM": circuit.qasm()}

        return circuit


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

    def compile(self, circuits, device="simulator", shots=1024, max_credits=3, basis_gates=None, coupling_map=None):
        """ Compile unrole the code
        circuits are circuits to unroll
        basis_gates are the base gates by default are: u1,u2,u3,cx,id
        """
        # TODO: Control device names
        qasm_circuits = []
        to_execute = {}
        for circuit in circuits:
            qasm_source, circuit_unrolled = self.unroller_code(self.__circuits['circuits'][circuit]["object"], basis_gates)
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
            #TODO: add timestamp, compilation
            self.__circuits['circuits'][circuit]["execution"] = {device: {"QASM_compile": qasm_source,
                                                              "compiled_circuits": circuit_unrolled,
                                                              "shots": shots,
                                                              "max_credit": max_credits,
                                                              "coupling_map": coupling_map,
                                                              "basis_gates": basis_gates}}
            if not device in to_execute:
                to_execute[device] = []
            to_execute[device].append({"circuit":circuit, "device":device})
        #TODO: improve ontly return the compiled circuits.

        self.__circuits["to_execute"] = to_execute
        return self.__circuits

    def run(self, wait=5, timeout=60):
        """Run a program (a pre compiled quantum program).
        wait time to check if the job is Completed.
        timeout time after that the execution stop
        """
        for backend in self.__circuits["to_execute"]:
            jobs = []
            for circuit in backend:
                jobs.append()

            print("backend that is running %s" % (backend))
            if backend in self.__online_devices:
                output = self.__api.run_job(self.__qasm_compile['compiled_circuits'],
                                            backend,
                                            self.__qasm_compile['shots'],
                                            self.__qasm_compile['max_credits'])
                # print(output)
                if 'error' in output:
                    return {"status": "Error", "result": output['error']}

                job_result = self.wait_for_job(output['id'],
                                               wait=wait,
                                               timeout=timeout)

                if job_result['status'] == 'Error':
                    return job_result
                self.__qasm_compile['status'] = job_result['status']
                self.__qasm_compile['used_credits'] = job_result['usedCredits']
            else:
                if backend == 'local_qasm_simulator':
                    job_result = self.run_local_qasm_simulator()
                    self.__qasm_compile['status'] = job_result['status']
                elif backend == 'local_unitary_simulator':
                    job_result = self.run_local_unitary_simulator()
                    self.__qasm_compile['status'] = job_result['status']
                else:
                    return {"status": "Error", "result": "Not local simulations"}

            self.__qasm_compile['compiled_circuits'] = job_result['qasms']

            # print(self.__qasm_compile['compiled_circuits'])
            self.__qasm_compile['status'] = job_result['status']

            return self.__qasm_compile

    def run_local_qasm_simulator(self):
        """run_local_qasm_simulator, run a program (precompile of quantum circuits).
        """
        shots = self.__qasm_compile['shots']
        qasms = self.__qasm_compile['compiled_circuits']

        outcomes = {'qasms':[]}
        for qasm_circuit in qasms:
            basis = []
            unroller = unroll.Unroller(qasm.Qasm(data=qasm_circuit['qasm']).parse(), SimulatorBackend(basis))
            unroller.backend.set_trace(False)
            unroller.execute()
            result = []
            if shots == 1:
                qasm_circuit = QasmSimulator(unroller.backend.circuit, random.random()).run()
            else:
                for i in range(shots):
                    b = QasmSimulator(unroller.backend.circuit, random.random()).run()
                    result.append(bin(b['result']['data']['classical_state'])[2:].zfill(b['number_of_cbits']))
                qasm_circuit["result"] = {"data":{"counts":dict(Counter(result))}}
            outcomes['qasms'].append(qasm_circuit)
        outcomes['status'] = 'COMPLETED'
        return outcomes

    def run_local_unitary_simulator(self):
        """run_local_qasm_simulator, run a program (precompile of quantum circuits).
        """
        shots = self.__qasm_compile['shots']
        qasms = self.__qasm_compile['compiled_circuits']
        basis = []
        outcomes = {'qasms':[]}
        for qasm_circuit in qasms:

            unroller = unroll.Unroller(qasm.Qasm(data=qasm_circuit['qasm']).parse(),
                                       SimulatorBackend(basis))
            unroller.backend.set_trace(False)  # print calls as they happen
            unroller.execute()  # Here is where simulation happens
            result = UnitarySimulator(unroller.backend.circuit).run()
            qasm_circuit["result"] = {"data":{"unitary":result}}
            outcomes['qasms'].append(qasm_circuit)
        outcomes['status'] = 'COMPLETED'
        return outcomes

    def execute(self, circuits, device, shots=1024,
                max_credits=3, wait=5, timeout=60, basis_gates=None, coupling_map=None):
        """Execute, compile and run a program (array of quantum circuits).
        program is a list of quantum_circuits
        api the api for the device
        device is a string for real or simulator
        shots is the number of shots
        max_credits is the credits of the experiments.
        basis_gates are the base gates by default are: u1,u2,u3,cx,id
        """
        self.compile(circuits, device, shots, max_credits,
                     basis_gates, coupling_map)
        output = self.run(wait, timeout)
        if 'error' in output:
            return {"status": "Error", "result": output['error']}

        return output

    def program_to_text(self, circuits=None):
        """Print a program (array of quantum circuits).

        program is a list of quantum circuits, if it's emty use the internal circuits
        """
        if not circuits:
            circuits = self.__circuits
        # TODO: Store QASM per circuit
        jobs = ""
        for name, circuit in circuits.items():
            circuit_name = "# Circuit: "+ name + "\n"
            qasm_source, circuit = self.unroller_code(circuit)
            jobs = jobs + circuit_name + qasm_source + "\n\n"
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

    def average_data(self, i, observable):
        """Compute the mean value of an diagonal observable.

        Takes in the data counts(i) and a corresponding observable in dict
        form and calculates sum_i value(i) P(i) where value(i) is the value of
        the observable for the i state.
        """
        counts = self.get_counts(i)
        temp = 0
        tot = sum(counts.values())
        for key in counts:
            if key in observable:
                temp += counts[key] * observable[key] / tot
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
            pass
            # basicplotter.plot_qsphere(data, circuit_number)

    def get_qasm_image(self, circuit):
        """Get imagen circuit representation from API."""
        pass

    def get_circuit(self, name):
        """get the circut by name.
        name of the circuit"""
        if name in self.__circuits["circuits"]:
            return self.__circuits["circuits"][name]
        else:
            return  {"status": "Error", "result": 'circuit not found'}

    def get_result(self, name, device):
        """get the get_result from one circut and backend
        name of the circuit
        device that use to compile, run or execute
        """
        if name in self.__circuits["circuits"] and device in self.__circuits["circuits"][name]['execution']:
            return self.__circuits["circuits"][name]['execution'][device]
        else:
            return  {"status": "Error", "result": 'circuit not found'}

    def get_data(self, results, name):
        """Get the dict of labels and counts from the output of get_job.
        results are the list of results
        name is the name or index of one circuit.
        NOTE: now only work with the index, we need need allow to access by the circuit name"""
        if not isinstance(name, str):
            return results['compiled_circuits'][name]['result']['data']['counts']
        else:
            pass

    #TODO: change the index for name and i think there is no point to get data above
    # ALSO i think we need an error if there is no results when we use a name

    def get_counts(self, name):
        """Get the dict of labels and counts from the output of get_job.
        name is the name or index of one circuit.
        NOTE: now only work with the index, we need need allow to access by the circuit name"""
        if not isinstance(name, str):
            return self.__qasm_compile['compiled_circuits'][name]['result']['data']['counts']
        else:
            pass
            # for qasm in self.__qasm_compile['compiled_circuits']:
            #     if 'result' not in self.__qasm_compile['compiled_circuits'][name]:
            #         raise QISKitException("the results have not been run")
            #     else:
            #         return self.__qasm_compile['compiled_circuits'][name]['result']['data']['counts']
