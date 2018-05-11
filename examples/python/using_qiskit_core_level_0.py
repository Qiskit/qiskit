# -*- coding: utf-8 -*-

# Copyright 2018 IBM RESEARCH. All Rights Reserved.
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
Example showing how to use QISKit at level 0 (novice).

See level 1 if you would like to understand how to compile

Note: if you have only cloned the QISKit repository but not
used `pip install`, the examples only work from the root directory.
"""

import time

# Import the QISKit modules
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, QISKitError
from qiskit import available_backends, execute, register, get_backend

try:
    import Qconfig
    register(Qconfig.APItoken, Qconfig.config['url'])
except:
    print("""WARNING: There's no connection with the API for remote backends.
             Have you initialized a Qconfig.py file with your personal token?
             For now, there's only access to local simulator backends...""")


def lowest_pending_jobs():
    """Returns the backend with lowest pending jobs."""
    list_of_backends = available_backends(
        {'local': False, 'simulator': False})
    device_status = [get_backend(backend).status
                     for backend in list_of_backends]

    best = min([x for x in device_status if x['available'] is True],
               key=lambda x: x['pending_jobs'])
    return best['name']


try:
    # Create a Quantum and Classical Register.
    qubit_reg = QuantumRegister(2)
    clbit_reg = ClassicalRegister(2)

    # making first circuit: bell state
    qc1 = QuantumCircuit(qubit_reg, clbit_reg)
    qc1.h(qubit_reg[0])
    qc1.cx(qubit_reg[0], qubit_reg[1])
    qc1.measure(qubit_reg, clbit_reg)

    # making another circuit: superpositions
    qc2 = QuantumCircuit(qubit_reg, clbit_reg)
    qc2.h(qubit_reg)
    qc2.measure(qubit_reg, clbit_reg)

    # setting up the backend
    print("(Local Backends)")
    print(available_backends({'local': True}))

    # runing the job
    job_sim = execute([qc1, qc2], "local_qasm_simulator")
    sim_result = job_sim.result()

    # Show the results
    print("simulation: ", sim_result)
    print(sim_result.get_counts(qc1))
    print(sim_result.get_counts(qc2))

    # see a list of available remote backends
    print("\n(Remote Backends)")
    print(available_backends({'local': False}))

    # Compile and run on a real device backend
    try:
        # select least busy available device and execute.
        best_device = lowest_pending_jobs()
        print("Running on current least busy device: ", best_device)

        # running the job
        job_exp = execute([qc1, qc2], backend=best_device, shots=1024, max_credits=10)

        print('JOB ID: {}'.format(job_exp.status['job_id']))
        lapse = 0
        interval = 10
        while not job_exp.done:
            print('Status @ {} seconds'.format(interval * lapse))
            print(job_exp.status)
            time.sleep(interval)
            lapse += 1
        print(job_exp.status)
        exp_result = job_exp.result()

        # Show the results
        print("experiment: ", exp_result)
        print(exp_result.get_counts(qc1))
        print(exp_result.get_counts(qc2))
    except:
        print("All devices are currently unavailable.")
except QISKitError as ex:
    print('There was an error in the circuit!. Error = {}'.format(ex))
