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

# Stable Modules
from . import QuantumRegister
from . import ClassicalRegister
from . import QuantumCircuit
from . import QISKitException

# Beta Modules
from . import unroll
from . import qasm
from . import mapper

# Local Simulator Modules
from . import simulators

import sys
sys.path.append("..")
import qiskit.extensions.standard

#TODO_ISMEAL_QUESATAION: I THINK THE FUNCTIONS NEED TO BE MOVED AROUND
# intilzers, seters, getters, runners


class QuantumProgram(object):
    """ Quantum Program Class

     Class internal properties """

    #TODO_ISMEAL_QUESATAION: I THINK we should remove IBMQX5qv2 i know it works but this forces us to
    # go down the path that we are working towards. I would add "real" back and give a comment that
    # real is the default real chip device which may change over time and the user will need to look up which
    # device is the real keyword from our devices tab(which i know does not exist yet)
    # Also for now ibmqx_qasm_simulator should defaul to the current simulator online which is what
    # simulator does as well but  my goal for it is to be an updated version of chirs's code when
    # he is finished  and then simulator can become our default like real
    __online_devices = ["IBMQX5qv2", "ibmqx2", "ibmqx3", "simulator", "ibmqx_qasm_simulator"]
    __local_devices = ["local_unitary_simulator", "local_qasm_simulator"]

    #TODO_ISMEAL_QUESATAION: CAN WE REMOVE THE COMMENTS BELOW
    # __specs = {}
    # __quantum_registers = {}
    # __classical_registers = {}
    # __quantum_program = {}
    __api = {}
    __api_config = {}
    # __qprogram = {}
    # __last_device_backend = ""
    # __to_execute = {}

    """
    Elements that are not python identifiers or string constants are denoted
    by "--description (type)--". For example, a circuit's name is denoted by
    "--circuit name (string)--" and might have the value "teleport".

    __quantum_program = {
        "circuits": {
            --circuit name (string)--: {
                "circuit": --circuit object (TBD)--,
                "qasm": --output of .qasm() (string)--,
                "execution": {  #### FILLED IN AFTER RUN -- JAY WANTS THIS MOVED DOWN ONE LAYER ####
                    --device name (string)--: {
                        "qasm_compiled": --compiled qasm output of .qasm() (string) after compliing--,
                        "coupling_map": --adjacency list (dict)--,
                        "basis_gates": --comma separated gate names (string)--,
                        "compiled_circuit": --local simulator input (dict that could be a JSON file) or None--,
                        "shots": --shots (int)--,
                        "max_credits": --credits (int)--,
                        "result": {
                            "data": {  #### DATA CAN BE A DIFFERENT DICTIONARY FOR EACH BACKEND ####
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
                "qasm_compiled": --compiled qasm output of .qasm() (string) after compliing--,
                "coupling_map": --adjacency list (dict)--,
                "basis_gates": --comma separated gate names (string)--,
                "compiled_circuit": --local simulator input (dict that could be a JSON file) or None--,
                "shots": --shots (int)--,
                "max_credits": --credits (int)--
                "seed": --initial seed for the simulator (int) --
            },
            ...
        ]
    }
    """
    # FUTURE IMPROVEMENTS (NOT NOW)
    # TODO. JAY: I DONT THINK coupling_map, basis_gates is needed in the __to_execute
    # TODO. JAY: I DONT THINK "qasm" is needed in the __quantum_prgram as
    # with the get_circuit_qasm method you can make it
    # TODO: JAY qasm_compiled and compiled_circuit are redundent if we do correctly.
    # THEY are the same thing for the different backends and currenlty we hope
    # (make them) stay consistant.
    # qasm_compiled is used by the API for online stuff.
    # compiled_circuit is used by my simulators. Ideally I would like
    # compiled_circuit to be a JSON FILE  which is very similar to the
    # output of the unroll SimulatorBackend and is a COMPLETE REPRESENTATION
    # of a circuit. It is this that is passed to the API or the simulator.
    #
    # A hack for this is (LETS not do this)
    #
    # delete qasm_compiled and then make compiled_circuit what qasm_compiled is
    # now. Then for the API it just gets passed compiled_circuit and for the
    # local simulator(s) we would do
    #   unroller = unroll.Unroller(qasm.Qasm(compiled_circuit).parse(), SimulatorBackend(basis_gates))
    #   unroller.backend.set_trace(False)
    #   unroller.execute()
    # and pass unroller.backend.circuit to the local simulators


    def __init__(self, specs=None, name=""):
        self.__quantum_program  = {"circuits":{}} #TODO_ISMEAL_QUESATAION: I CHANGE __CIRCUITS TO __quantum_program
        self.__quantum_registers = {}
        self.__classical_registers = {}
        self.__init_circuit = None
        self.__name = name #TODO_ISMEAL_QUESATAION: IS THIS USED
        self.__qprogram = {} #TODO_ISMEAL_QUESATAION: IS THIS USED __quantum_program  has replaced it
        self.__last_device_backend = ""
        self.__to_execute = {}

        self.mapper = mapper

        if specs:
            self.__init_specs(specs)

    def quantum_elements(self, specs=None):
        """Return the basic elements, Circuit, Quantum Registers, Classical Registers"""
        #TODO_ISMEAL_QUESATAION: I PREFER GET_* for getters
        if not specs:
            specs = self.get_specs()

        return self.__init_circuit, \
            self.__quantum_registers[list(self.__quantum_registers)[0]], \
            self.__classical_registers[list(self.__classical_registers)[0]]

    def quantum_registers(self, name):
        #TODO_ISMEAL_QUESATAION: I PREFER GET_* for getters
        """Return a specific Quantum Registers"""
        return self.__quantum_registers[name]

    def classical_registers(self, name):
        #TODO_ISMEAL_QUESATAION: I PREFER GET_* for getters
        """Return a specific Classical Registers"""
        return self.__classical_registers[name]

    def circuit(self, name):
        #TODO_ISMEAL_QUESATAION: I PREFER GET* for getters and this should be get_quantum_circuit.
        """Return a specific Circuit Object"""
        return self.__quantum_program['circuits'][name]['circuit']

    def get_specs(self):
        """Return the program specs"""
        return self.__specs

    def api_config(self):
        """Return the program specs"""
        #TODO_ISMEAL_QUESATAION: I PREFER GET_* and really doc needs work
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

        circuit_object = qasm.Qasm(filename=qasm_file).parse()

        self.__quantum_program['circuits'][name] = {"circuit": circuit_object, "qasm": circuit_object.qasm()}

        #TODO_ISMEAL_QUESATAION: WHY DO WE NEED TO RETURN SOMETHING
        return self.__quantum_program['circuits'][name]


    def unroller_code(self, circuit, basis_gates=None):
        """ Unroll the code
        circuit are circuits to unroll
        basis_gates are the base gates by default are: u1,u2,u3,cx,id
        """
        if not basis_gates:
            basis_gates = "u1,u2,u3,cx,id"  # QE target basis

        unrolled_circuit = unroll.Unroller(qasm.Qasm(data=circuit.qasm()).parse(),
                                           unroll.CircuitBackend(basis_gates.split(",")))
        unrolled_circuit.execute()

        circuit_unrolled = unrolled_circuit.backend.circuit  # circuit DAG
        qasm_source = circuit_unrolled.qasm(qeflag=True)
        return qasm_source, circuit_unrolled

    def compile(self, name_of_circuits, device="simulator", shots=1024, max_credits=3, basis_gates=None, coupling_map=None, seed=None):
        """Compile the name_of_circuits by names.

        name_of_circuits is a list of circuit names to compile.
        device is the target device name.
        basis_gates are the base gates by default are: u1,u2,u3,cx,id
        coupling_map is the adjacency list for coupling graph

        This method adds elements of the following form to the self.__to_execute list corresponding to the device:

        --device name (string)--: [
                {
                    "name": --circuit name (string)--,
                    "qasm_compiled": --compiled qasm (string)--,
                    "coupling_map": --adjacency list (dict)--,
                    "basis_gates": --comma separated gate names (string)--,
                    "compiled_circuit": --local simulator input (dict) OR None--,
                    "shots": --shots (int)--,
                    "max_credits": --credits (int)--
                    "seed": --initial seed for the simulator (int) --
                },
                ...
            ]
        }
        """
        # TODO: Control device names
        if name_of_circuits == []:
            return {"status": "Error", "result": 'Not circuits'}

        for name in name_of_circuits:
            # TODO: The circuit object has to have .qasm() method; currently several different types
            qasm_compiled, dag_unrolled = self.unroller_code(self.__quantum_program['circuits'][name]['circuit'], basis_gates)
            if coupling_map:
                print("pre-mapping properties: %s"
                      % dag_unrolled.property_summary())
                # Insert swap gates
                coupling = self.mapper.Coupling(coupling_map)
                dag_unrolled, final_layout = self.mapper.swap_mapper(
                    dag_unrolled, coupling)
                print("layout: %s" % final_layout)
                # Expand swaps
                qasm_compiled, dag_unrolled = self.unroller_code(
                    dag_unrolled)
                # Change cx directions
                dag_unrolled = mapper.direction_mapper(dag_unrolled,
                                                       coupling)
                # Simplify cx gates
                mapper.cx_cancellation(dag_unrolled)
                # Simplify single qubit gates
                dag_unrolled = mapper.optimize_1q_gates(dag_unrolled)
                qasm_compiled = dag_unrolled.qasm(qeflag=True)
                print("post-mapping properties: %s"
                      % dag_unrolled.property_summary())
            # TODO: add timestamp, compilation
            if device not in self.__to_execute:
                self.__to_execute[device] = []
            # We overwrite this data on each compile. A user would need to make the same circuit
            # with a different name for this not to be the case.

            self.__quantum_program["circuits"][name]["qasm"] = self.__quantum_program['circuits'][name]['circuit'].qasm()
            #TODO_ISMEAL_QUESATAION: I THINK THIS SHOULD MAKE A NEW EXECUTION iF
            #IT DOES NOT EXIST AND IF IT DOES IT SHOULD ONLY OVERRIDER THE BACKEND
            self.__quantum_program["circuits"][name]["execution"] = {}
            job = {}
            job["name"] = name
            job["qasm_compiled"] = qasm_compiled
            job["coupling_map"] = coupling_map
            job["basis_gates"] = basis_gates
            job["compiled_circuit"] = None
            job["shots"] = shots
            job["max_credits"] = max_credits
            job["seed"] = None
            if device in ["local_unitary_simulator", "local_qasm_simulator"]:
                unroller = unroll.Unroller(qasm.Qasm(data=qasm_compiled).parse(), SimulatorBackend(basis_gates))
                unroller.backend.set_trace(False)
                unroller.execute()
                job["compiled_circuit"] = unroller.backend.circuit
                job["seed"] = random.random()
                if seed is not None:
                    job["seed"] = seed
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
                for job in self.__to_execute[backend]:
                    jobs.append({'qasm': job["qasm_compiled"]})
                    shots = job["shots"]
                    max_credits = job["max_credits"]
                    if last_shots == -1:
                        last_shots = shots
                    else:
                        if last_shots != shots:
                            return {"status": "Error", "result":'Online devices only support job batches with equal numbers of shots'}
                    if last_max_credits == -1:
                        last_max_credits = max_credits
                    else:
                        if last_max_credits != max_credits:
                            return  {"status": "Error", "result":'Online devices only support job batches with equal max credits'}

                print("running on backend: %s" % (backend))
                output = self.__api.run_job(jobs, backend, last_shots, last_max_credits)
                if 'error' in output:
                    return {"status": "Error", "result": output['error']}
                job_result = self.wait_for_job(output['id'], wait=wait, timeout=timeout)

                if job_result['status'] == 'Error':
                    return job_result
            else:
                jobs = []
                for job in self.__to_execute[backend]:
                    #TODO_JAY: PUT SEED IN JOB
                    jobs.append({"compiled_circuit": job["compiled_circuit"], "shots": job["shots"]})
                print("running on backend: %s" % (backend))
                if backend == "local_qasm_simulator":
                    job_result = self.run_local_qasm_simulator(jobs)
                elif backend == "local_unitary_simulator":
                    job_result = self.run_local_unitary_simulator(jobs)
                else:
                    return {"status": "Error", "result": 'Not a local simulator'}

            assert len(self.__to_execute[backend]) == len(job_result["qasms"]), "Internal error in QuantumProgram.run(), job_result"

            # Fill data into self.__quantum_program for this backend
            index = 0
            for job in self.__to_execute[backend]:
                name = job["name"]
                if name not in self.__quantum_program["circuits"]:
                    return {"status": "Error", "result": "Internal error, circuit not found"}
                if backend not in self.__quantum_program["circuits"][name]["execution"]:
                    self.__quantum_program["circuits"][name]["execution"][backend] = {}
                # TODO: return date, executionId, ...
                for field in ["qasm_compiled", "coupling_map", "basis_gates", "compiled_circuit", "shots", "max_credits"]:
                    self.__quantum_program["circuits"][name]["execution"][backend][field] = job[field]
                self.__quantum_program["circuits"][name]["execution"][backend]["result"] = job_result["qasms"][index]["result"]
                self.__quantum_program["circuits"][name]["execution"][backend]["status"] = job_result["qasms"][index]["status"]
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
                #TODO_JAY: PUT SEED IN JOB
                qasm_circuit = simulators.QasmSimulator(job["compiled_circuit"], random.random()).run()
                one_result["result"] = qasm_circuit["result"]
                one_result["status"] = 'COMPLETED'
            else:
                result = []
                for i in range(job["shots"]):
                    b = simulators.QasmSimulator(job["compiled_circuit"], random.random()).run()
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
            unitary_circuit = simulators.UnitarySimulator(job["compiled_circuit"]).run()
            one_result["result"] = unitary_circuit["result"]
            one_result["status"] = 'COMPLETED'
            job_results['qasms'].append(one_result)
        return job_results

    def execute(self, name_of_circuits, device, shots=1024,
                max_credits=3, wait=5, timeout=60, basis_gates=None, coupling_map=None, seed=None):
        """Execute, compile and run a program (array of quantum circuits).
        program is a list of quantum_circuits
        api the api for the device
        device is a string for real or simulator
        shots is the number of shots
        max_credits is the credits of the experiments.
        basis_gates are the base gates by default are: u1,u2,u3,cx,id
        """
        self.compile(name_of_circuits, device, shots, max_credits,
                     basis_gates, coupling_map, seed)
        output = self.run(wait, timeout)
        return output

    def program_to_text(self, circuits=None):
        """Print a program (array of quantum circuits).

        program is a list of quantum circuits, if it's emty use the internal circuits
        """

        #TODO_ISMEAL_QUESATAION: I WANT TO KILL THIS FUNCTION and have
        # get_circuit_qasm and git_circut_qasms SEE BELOW
        if not circuits:
            circuits = self.__quantum_program['circuits']
        # TODO: Store qasm per circuit
        jobs = ""
        for name, circuit in circuits.items():
            circuit_name = "# Circuit: "+ name + "\n"
            qasm_source, circuit = self.unroller_code(circuit['circuit'])
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
                return {"status": "Error", "result": "Time Out"}
            time.sleep(wait)
            timer += wait
            print("status = %s (%d seconds)" % (job['status'], timer))
            job = self.__api.get_job(jobid)
            if job['status'] == 'ERROR_CREATING_JOB' or job['status'] == 'ERROR_RUNNING_JOB':
                return {"status": "Error", "result": job['status']}

        # Get the results
        return job

    def average_data(self, name, observable):
        """Compute the mean value of an diagonal observable.

        Takes in an obserbaleobservable in dict
        form and calculates sum_i value(i) P(i) where value(i) is the value of
        the observable for the i state

        returns a double
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
        #TODO_ISMEAL_QUESATAION: do we need to fill in the qasm my philophy is this
        #is not filled until we compile or save when we have a save and why do we return
        self.__quantum_program['circuits'][name] = {"name":name, "circuit": circuit_object, "qasm": circuit_object.qasm()}

        return circuit_object

    def create_circuit(self, name, qregisters=None, cregisters=None, circuit_object=None):
        """Create a new Quantum Circuit into the Quantum Program
        name is a string, the name of the circuit
        qregisters is a Array of Quantum Registers, can be String, by name or the object reference
        cregisters is a Array of Classical Registers, can be String, by name or the object reference
        """
        #TODO_ISMEAL_QUESATAION: do we need to fill in the qasm my philosophy is this
        #is not filled until we compile or save when we have a save and why do we return
        if not qregisters:
            qregisters = []
        if not cregisters:
            cregisters = []

        if not circuit_object:
            circuit_object = QuantumCircuit()
        self.__quantum_program['circuits'][name] = {"name":name, "circuit": circuit_object, "qasm": circuit_object.qasm()}

        for register in qregisters:
            if isinstance(register, str):
                self.__quantum_program['circuits'][name]['circuit'].add(self.__quantum_registers[register])
            else:
                self.__quantum_program['circuits'][name]['circuit'].add(register)
        for register in cregisters:
            if isinstance(register, str):
                self.__quantum_program['circuits'][name]['circuit'].add(self.__classical_registers[register])
            else:
                self.__quantum_program['circuits'][name]['circuit'].add(register)

        self.__quantum_program['circuits'][name]['qasm'] = self.__quantum_program['circuits'][name]['circuit'].qasm()

        return self.__quantum_program['circuits'][name]['circuit']

    def create_quantum_registers(self, name, size):
        """Create a new set of Quantum Registers"""
        self.__quantum_registers[name] = QuantumRegister(name, size)
        print(">> quantum_registers created:", name, size)
        return self.__quantum_registers[name]

    #TODO_ISMEAL_QUESATAION: HOW IS THIS DIFFERENT TO ABOVE
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

    #TODO_ISMEAL_QUESATAION:  SHOULD WE USE THE SAME NAME REG->registers
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
        """Get image circuit representation from API."""
        pass

    #TODO_ISMEAL_QUESATAION: HOW IS THIS CURRENTLY USED.
    #I THINK THIS SHOLD BE get_circuits_qasm and then we
    #remove program_to_text and then you can have get_circuit_qasms that
    #takes in and array of names and self.__quantum_program['circuits'][name]["qasm"]
    #WHICH MAY BE EMPYT AND IF SO YOU NEED TO RUN
    #self.__quantum_program["circuits"][name]["qasm"] = self.__quantum_program["circuits"][name]['circuit'].qasm()
    # AND THEN RETURN
    def get_circuit(self, name):
        """get the circut by name.
        name of the circuit"""
        if name in self.__quantum_program['circuits']:
            return self.__quantum_program['circuits'][name]
        else:
            return  {"status": "Error", "result": 'Circuit not found'}

    def get_result(self, name, device=None):
        """get the get_result from one circut and backend
        name of the circuit
        device that use to compile, run or execute
        """
        if not device:
            device = self.__last_device_backend

        if name in self.__quantum_program["circuits"]:
            return self.__quantum_program["circuits"][name]['execution'][device]['result']
        else:
            return  {"status": "Error", "result": 'Circuit not found'}

    def get_data(self, name, device=None):
        """Get the dict of labels and counts from the output of get_job.
        results are the list of results
        name is the name or index of one circuit."""
        if not device:
            device = self.__last_device_backend
        return self.__quantum_program["circuits"][name]['execution'][device]['result']['data']

    #TODO_ISMEAL_QUESATAION: I THINK get_counts, get_data, and get result need to be
    #nested get_counts calls _get_data and get_data calls get_results
    #and use the try as in get_counts. change the index for name and i think there is no point to get data above
    # ALSO i think we need an error if there is no results when we use a name
    def get_counts(self, name, device=None):
        """Get the dict of labels and counts from the output of get_job.
        name is the name or index of one circuit."""
        if not device:
            device = self.__last_device_backend
        # TODO: check that there are results
        try:
            return self.__quantum_program["circuits"][name]['execution'][device]['result']['data']['counts']
        except KeyError:
            return  {"status": "Error", "result": 'Error in circuit name'}
