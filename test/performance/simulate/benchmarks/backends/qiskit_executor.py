"""
Qiskit Executor
"""
import os
import time
import json
from qiskit import register, execute


class QiskitExecutor(object):
    """
    Qiskit Executor
    """
    def __init__(self, executor):
        self.name = ['local_qasm_simulator', 'local_clifford_simulator',
                     'local_statevector_simulator', 'local_unitary_simulator',
                     'ibmq_qasm_simulator', 'ibmqx2', 'ibmqx4',
                     'QS1_1', 'ibmq_20_tokyo']
        self.seed = executor.seed
        self.application = executor.name
        self.backend_name = executor.backend_name
        self.result = None
        self.filename = None

    def run_simulation(self, qcirc):
        """
        Execute Simulator
        """
        if self.backend_name.startswith("ibmq"):
            import Qconfig
            register(Qconfig.APItoken, **Qconfig.config)
        elif not self.backend_name.startswith("local"):
            raise Exception('only ibmqx or local simulators are supported')

        start = time.time()
        job_sim = execute(qcirc, backend=self.backend_name, shots=1,
                          max_credits=5, hpc=None, seed=self.seed)
        ret = job_sim.result()
        elapsed = time.time() - start

        if not ret.get_circuit_status(0) == "DONE":
            return -1.0

        if self.backend_name.startswith("ibmqx"):
            elapsed = ret.get_data(None)["time"]

        self.result = ret.get_counts(None)
        return elapsed

    def verify_result(self, depth=0, qubit=0):
        """
        Verify Simulation Result
        """
        if not self.filename:
            if depth > 0:
                self.filename = self.application + "_n" + str(qubit) + "_d" + str(depth)
            else:
                self.filename = self.application + "_n" + str(qubit)

        if not os.path.exists("ref/" + self.application):
            raise Exception("Verification not support for " + self.application)

        ref_file_name = "ref/"+self.application + "/" + \
                        os.path.basename(self.filename)+"."+self.backend_name+".ref"
        if not os.path.exists(ref_file_name):
            raise Exception("Reference file not exist: " + ref_file_name)

        ref_file = open(ref_file_name)
        ref_data = ref_file.read()
        ref_file.close()
        ref_data = json.loads(ref_data)
        sim_result_keys = self.result.keys()

        for key in sim_result_keys:
            if key not in ref_data:
                raise Exception(key + " not exist in " + ref_file_name)
            ref_count = ref_data[key]
            count = self.result[key]

            if ref_count != count:
                raise Exception(" Count is differ: " + str(count) +
                                " and " + str(ref_count))
