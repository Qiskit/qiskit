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
    __qprogram = {}
    __last_device_backend = ""
    __to_execute = {}

    """
    Elements that are not python identifiers or string constants are denoted
    by "--description (type)--". For example, a circuit's name is denoted by
    "--circuit name (string)--" and might have the value "teleport".

    __circuits = {
        "circuits": {
            --circuit name (string)--: {
                "name": --circuit name (string)--,
                "object": --circuit object (TBD)--,
                "QASM": --output of .qasm() (string)--,
                "execution": {  # FILLED IN AFTER RUN
                    --device name (string)--: {
                        "QASM_compiled": --compiled QASM (string)--,
                        "coupling_map": --adjacency list (dict)--,
                        "basis_gates": --comma separated gate names (string)--,
                        "compiled_circuit": --local simulator input (dict) or None--,
                        "shots": --shots (int)--,
                        "max_credits": --credits (int)--,
                        "result": {
                            "data": {  # DATA IS A DIFFERENT OBJECT FOR EACH SIMULATOR
                                "counts": {’00000’: XXXX, ’00001’: XXXXX},
                                "time"  : xx.xxxxxxxx
                            },
                            "date"  : "2017−05−09Txx:xx:xx.xxxZ",
                            "status": --status (string)--
                        }
                    },
            }
        }
    }

    __to_execute = {
        --device name (string)--: [
            {
                "name": --circuit name (string)--,
                "QASM_compiled": --compiled QASM (string)--,
                "coupling_map": --adjacency list (dict)--,
                "basis_gates": --comma separated gate names (string)--,
                "compiled_circuit": --local simulator input (dict) OR None--,
                "shots": --shots (int)--,
                "max_credits": --credits (int)--
            },
            ...
        ]
    }
    """

    def __init__(self, specs=None, name=""):
        self.__circuits = {"circuits":{}}
        self.__quantum_registers = {}
        self.__classical_registers = {}
        self.__init_circuit = None
        self.__name = name
        self.__qprogram = {}
        self.__last_device_backend = ""

        self.mapper = mapper

        if specs:
            self.__init_specs(specs)

    def quantum_elements(self, specs=None):
        """Return the basic elements, Circuit, Quantum Registers, Classical Registers"""
        if not specs:
            specs = self.get_specs()

        return self.__init_circuit, \
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
        return self.__circuits['circuits'][name]['object']

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
        """Compile the circuits.

        circuits is a list of circuit names to compile.
        device is the target device name.
        basis_gates are the base gates by default are: u1,u2,u3,cx,id
        coupling_map is the adjacency list for coupling graph

        This method adds elements of the following form to the self.__to_execute list corresponding to the device:

        --device name (string)--: [
                {
                    "name": --circuit name (string)--,
                    "QASM_compiled": --compiled QASM (string)--,
                    "coupling_map": --adjacency list (dict)--,
                    "basis_gates": --comma separated gate names (string)--,
                    "compiled_circuit": --local simulator input (dict) OR None--,
                    "shots": --shots (int)--,
                    "max_credits": --credits (int)--
                },
                ...
            ]
        }
        """
        # TODO: Control device names
        if circuits == []:
            return {"status": "Error", "result": 'Not circuits'}
            
        for circuit in circuits:
            # TODO: The circuit object has to have .qasm() method; currently several different types
            QASM_compiled, dag_unrolled = self.unroller_code(self.__circuits['circuits'][circuit]["object"], basis_gates)
            if coupling_map:
                print("pre-mapping properties: %s"
                      % dag_unrolled.property_summary())
                # Insert swap gates
                coupling = self.mapper.Coupling(coupling_map)
                dag_unrolled, final_layout = self.mapper.swap_mapper(
                    dag_unrolled, coupling)
                print("layout: %s" % final_layout)
                # Expand swaps
                QASM_compiled, dag_unrolled = self.unroller_code(
                    dag_unrolled)
                # Change cx directions
                dag_unrolled = mapper.direction_mapper(dag_unrolled,
                                                       coupling)
                # Simplify cx gates
                mapper.cx_cancellation(dag_unrolled)
                # Simplify single qubit gates
                dag_unrolled = mapper.optimize_1q_gates(dag_unrolled)
                QASM_compiled = dag_unrolled.qasm(qeflag=True)
                print("post-mapping properties: %s"
                      % dag_unrolled.property_summary())
            # TODO: add timestamp, compilation
            if device not in self.__to_execute:
                self.__to_execute[device] = []
            # We overwrite this data on each compile. A user would need to make the same circuit
            # with a different name for this not to be the case.
            self.__circuits["circuits"][circuit]["QASM"] = self.__circuits["circuits"][circuit]["object"].qasm()
            self.__circuits["circuits"][circuit]["execution"] = {}
            job = {}
            job["name"] = circuit
            job["QASM_compiled"] = QASM_compiled
            job["coupling_map"] = coupling_map
            job["basis_gates"] = basis_gates
            job["compiled_circuit"] = None
            job["shots"] = shots
            job["max_credits"] = max_credits
            if device in ["local_unitary_simulator", "local_qasm_simulator"]:
                unroller = unroll.Unroller(qasm.Qasm(data=QASM_compiled).parse(), SimulatorBackend(basis_gates))
                unroller.backend.set_trace(False)
                unroller.execute()
                job["compiled_circuit"] = unroller.backend.circuit
            self.__to_execute[device].append(job)
        return {"status": "COMPLETED", "result": 'all done'}

    def run(self, wait=5, timeout=60):
        """Run a program (a pre compiled quantum program).

        All input for run comes from self.__to_execute
        wait time to check if the job is Completed.
        timeout time after that the execution stop
        """
        for backend in self.__to_execute:
            self.__last_device_backend = backend
            if backend in self.__online_devices:
                last_shots = -1
                last_max_credits = -1
                jobs = []
                for circuit in self.__to_execute[backend]:
                    jobs.append({'qasm': circuit["QASM_compiled"]})
                    shots = circuit["shots"]
                    max_credits = circuit["max_credits"]
                    if last_shots == -1:
                        last_shots = shots
                    else:
                        if last_shots != shots:
                            return "Error: Online devices only support job batches with equal numbers of shots"
                    if last_max_credits == -1:
                        last_max_credits = max_credits
                    else:
                        if last_max_credits != max_credits:
                            return "Error: Online devices only support job batches with equal max credits"
                print("running on backend: %s" % (backend))
                output = self.__api.run_job(jobs, backend, last_shots, last_max_credits)
                if 'error' in output:
                    return "Error: " + output['error']
                job_result = self.wait_for_job(output['id'], wait=wait, timeout=timeout)

                if job_result['status'] == 'Error':
                    return "Error: " + job_result['result']
            else:
                jobs = []
                for circuit in self.__to_execute[backend]:
                    jobs.append({"compiled_circuit": circuit["compiled_circuit"], "shots": circuit["shots"]})
                if backend == "local_qasm_simulator":
                    job_result = self.run_local_qasm_simulator(jobs)
                elif backend == "local_unitary_simulator":
                    job_result = self.run_local_unitary_simulator(jobs)
                else:
                    return "Error: Not a local simulator"

            assert len(self.__to_execute[backend]) == len(job_result["qasms"]), "Internal error in QuantumProgram.run(), job_result"

            # Fill data into self.__circuits for this backend
            index = 0
            print(self.__to_execute[backend],'///////////////')
            for circuit in self.__to_execute[backend]:
                name = circuit["name"]
                if name not in self.__circuits["circuits"]:
                    return {"status": "Error", "result": "Internal error, circuit not found"}
                if backend not in self.__circuits["circuits"][name]["execution"]:
                    self.__circuits["circuits"][name]["execution"][backend] = {}
                # TODO: return date, executionId, ...
                for field in ["QASM_compiled", "coupling_map", "basis_gates", "compiled_circuit", "shots", "max_credits"]:
                    self.__circuits["circuits"][name]["execution"][backend][field] = circuit[field]
                self.__circuits["circuits"][name]["execution"][backend]["result"] = job_result["qasms"][index]["result"]
                self.__circuits["circuits"][name]["execution"][backend]["status"] = job_result["qasms"][index]["status"]
                index += 1

        # Clear the list of compiled programs to execute
        self.__to_execute = {}

        return  {"status": "COMPLETED", "result": 'all done'}

    def run_local_qasm_simulator(self, jobs):
        """run_local_qasm_simulator, run a program (precompile of quantum circuits).
        jobs is list of dicts {"compiled_circuit": simulator input data, "shots": integer num shots}

        returns
        job_results = {
            "qasms": [
                {
                    "result": DATA,
                    "status": DATA,
                },
                ...
            ]
        }
        """
        job_results = {"qasms": []}
        for job in jobs:
            one_result = {'result': None, 'status': "FAIL"}
            if job["shots"] == 1:
                # TODO: seed control
                qasm_circuit = QasmSimulator(job["compiled_circuit"], random.random()).run()
                one_result["result"] = qasm_circuit["result"]
                one_result["status"] = 'COMPLETED'
            else:
                result = []
                for i in range(job["shots"]):
                    b = QasmSimulator(job["compiled_circuit"], random.random()).run()
                    result.append(bin(b['result']['data']['classical_state'])[2:].zfill(b['number_of_cbits']))
                one_result["result"] = {"data": {"counts": dict(Counter(result))}}
                one_result["status"] = 'COMPLETED'
            job_results['qasms'].append(one_result)
        return job_results

    def run_local_unitary_simulator(self, jobs):
        """run_local_unitary_simulator, run a program (precompile of quantum circuits).
        jobs is list of dicts {"compiled_circuit": simulator input data}

        returns
        job_results = {
            "qasms": [
                {
                    "result": DATA,
                    "status": DATA,
                },
                ...
            ]
        }
        """
        job_results = {"qasms": []}
        for job in jobs:
            one_result = {'result': None, 'status': "FAIL"}
            unitary_circuit = UnitarySimulator(job["compiled_circuit"]).run()
            one_result["result"] = unitary_circuit["result"]
            one_result["status"] = 'COMPLETED'
            job_results['qasms'].append(one_result)
        return job_results

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
        return output

    def program_to_text(self, circuits=None):
        """Print a program (array of quantum circuits).

        program is a list of quantum circuits, if it's emty use the internal circuits
        """
        if not circuits:
            circuits = self.__circuits['circuits']
        # TODO: Store QASM per circuit
        jobs = ""
        for name, circuit in circuits.items():
            circuit_name = "# Circuit: "+ name + "\n"
            qasm_source, circuit = self.unroller_code(circuit['object'])
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

    def average_data(self, name, observable):
        """Compute the mean value of an diagonal observable.

        Takes in the data counts(i) and a corresponding observable in dict
        form and calculates sum_i value(i) P(i) where value(i) is the value of
        the observable for the i state.
        """
        counts = self.get_counts(name)
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
                self.__init_circuit = self.create_circuit(name=circuit["name"],
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

    def add_circuit(self, name, circuit_object):
        """Add anew circuit based in a Object representation.
        name is the name or index of one circuit."""

        self.__circuits['circuits'][name] = {"name":name, "object": circuit_object, "QASM": circuit_object.qasm()}

        return circuit_object

    def create_circuit(self, name, qregisters=None, cregisters=None, circuit_object=None):
        """Create a new Quantum Circuit into the Quantum Program
        name is a string, the name of the circuit
        qregisters is a Array of Quantum Registers, can be String, by name or the object reference
        cregisters is a Array of Classical Registers, can be String, by name or the object reference
        """
        if not qregisters:
            qregisters = []
        if not cregisters:
            cregisters = []

        if not circuit_object:
            circuit_object = QuantumCircuit()
        self.__circuits['circuits'][name] = {"name":name, "object": circuit_object, "QASM": circuit_object.qasm()}

        for register in qregisters:
            if isinstance(register, str):
                self.__circuits['circuits'][name]['object'].add(self.__quantum_registers[register])
            else:
                self.__circuits['circuits'][name]['object'].add(register)
        for register in cregisters:
            if isinstance(register, str):
                self.__circuits['circuits'][name]['object'].add(self.__classical_registers[register])
            else:
                self.__circuits['circuits'][name]['object'].add(register)

        self.__circuits['circuits'][name]['QASM'] = self.__circuits['circuits'][name]['object'].qasm()

        return self.__circuits['circuits'][name]['object']

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

    def plotter(self, name, device=None, method="histogram", number_to_keep=None):
        """Plot the results
        method: histogram/qsphere
        circuit: Print one circuit
        """
        data = self.get_counts(name, device)

        if method == "histogram":
            basicplotter.plot_histogram(data, number_to_keep)
        else:
            pass
            # TODO: only work with unitary simulator; basicplotter.plot_qsphere(data)

    def get_qasm_image(self, circuit):
        """Get imagen circuit representation from API."""
        pass

    def get_circuit(self, name):
        """get the circut by name.
        name of the circuit"""
        print (self.__circuits['circuits'],'<<<<<--------')
        if name in self.__circuits['circuits']:
            return self.__circuits['circuits'][name]
        else:
            return  {"status": "Error", "result": 'Circuit not found'}

    def get_result(self, name, device=None):
        """get the get_result from one circut and backend
        name of the circuit
        device that use to compile, run or execute
        """
        print(name,'<<<-') 
        print (self.__circuits['circuits'],'<<<<<--------')
        if not device:
            device = self.__last_device_backend

        if name in self.__circuits["circuits"]:
            return self.__circuits["circuits"][name]['execution'][device]['result']
        else:
            return  {"status": "Error", "result": 'Circuit not found'}

    def get_data(self, name, device=None):
        """Get the dict of labels and counts from the output of get_job.
        results are the list of results
        name is the name or index of one circuit."""
        if not device:
            device = self.__last_device_backend
        return self.__circuits["circuits"][name]['execution'][device]['result']['data']

    #TODO: change the index for name and i think there is no point to get data above
    # ALSO i think we need an error if there is no results when we use a name

    def get_counts(self, name, device=None):
        """Get the dict of labels and counts from the output of get_job.
        name is the name or index of one circuit."""
        if not device:
            device = self.__last_device_backend
        # TODO: check that there are results
        try:
            return self.__circuits["circuits"][name]['execution'][device]['result']['data']['counts']
        except KeyError:
            return  {"status": "Error", "result": 'Error in cicuit name'}
