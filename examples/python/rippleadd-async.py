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
Ripple adder example based on Cuccaro et al, quant-ph/0410184.
"""

import sys
import os
import time

# We don't know from where the user is running the example,
# so we need a relative position from this file path.
# TODO: Relative imports for intra-package imports are highly discouraged.
# http://stackoverflow.com/a/7506006
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from qiskit import QuantumProgram, QuantumCircuit

import Qconfig

online_backend = "ibmqx_qasm_simulator"
local_backend = "local_qasm_simulator"

# Whether we have connection with API servers or not. If not, we only launch
# jobs to the local simulator
offline = False
NUM_JOBS = 2 # TODO Parameterize
n = 2
QPS_SPECS = {
    "circuits": [
        {
            "name": "rippleadd",
            "quantum_registers": [
                {"name": "a",
                 "size": n},
                {"name": "b",
                 "size": n},
                {"name": "cin",
                 "size": 1},
                {"name": "cout",
                 "size": 1}
            ],
            "classical_registers": [
                {"name": "ans",
                 "size": n + 1},
            ]
        }
    ]
}

qp = QuantumProgram(specs=QPS_SPECS)
qc = qp.get_circuit("rippleadd")
a = qp.get_quantum_register("a")
b = qp.get_quantum_register("b")
cin = qp.get_quantum_register("cin")
cout = qp.get_quantum_register("cout")
ans = qp.get_classical_register("ans")


def majority(p, a, b, c):
    """Majority gate."""
    p.cx(c, b)
    p.cx(c, a)
    p.ccx(a, b, c)


def unmajority(p, a, b, c):
    """Unmajority gate."""
    p.ccx(a, b, c)
    p.cx(c, a)
    p.cx(a, b)


# Build a temporary subcircuit that adds a to b,
# storing the result in b
adder_subcircuit = QuantumCircuit(cin, a, b, cout)
majority(adder_subcircuit, cin[0], b[0], a[0])
for j in range(n - 1):
    majority(adder_subcircuit, a[j], b[j + 1], a[j + 1])
adder_subcircuit.cx(a[n - 1], cout[0])
for j in reversed(range(n - 1)):
    unmajority(adder_subcircuit, a[j], b[j + 1], a[j + 1])
unmajority(adder_subcircuit, cin[0], b[0], a[0])

# Set the inputs to the adder
qc.x(a[0])  # Set input a = 0...0001
qc.x(b)   # Set input b = 1...1111
# Apply the adder
qc += adder_subcircuit
# Measure the output register in the computational basis
for j in range(n):
    qc.measure(b[j], ans[j])
qc.measure(cout[0], ans[n])

###############################################################
# Set up the API and execute the program.
###############################################################
try:
    qp.set_api(Qconfig.APItoken, Qconfig.config["url"])
except:
    offline = True
    print("""WARNING: There's no connection with IBMQuantumExperience servers.
             cannot test I/O intesive tasks, will only test CPU intensive tasks
             running the jobs in the local simulator""")

qobjs = []
# Create online (so I/O bound) jobs if we have connetion or local (so CPU bound)
# jobs otherwise
if not offline:
    print("Creating %d online jobs..." % NUM_JOBS)
    for _ in range(0, NUM_JOBS):
        qobjs.append(qp.compile(["rippleadd"], backend=online_backend,
                                coupling_map=None, shots=1024))

print("Creating %d local jobs..." % NUM_JOBS)
# Create CPU intensive jobs
for _ in range(0, NUM_JOBS):
    qobjs.append(qp.compile(["rippleadd"], backend=local_backend,
                            coupling_map=None, shots=1024))

end = False
def print_results_callback(results, error=None):
    """This function will be called once all jobs have finished."""
    if error != None:
        print("There was an error executing the circuits!!: Error = {}".format(error))
        return

    for result in results:
        print("result: {}".format(result))
        try:
            print(result.get_counts("rippleadd"))
        except Exception as ex:
            print("ERROR: {}".format(ex))

        print("============")
    global end
    end = True

print("Running jobs asynchronously....")
# This call is asynchronous, it won't block!
qp.run_batch_async(qobjs, callback=print_results_callback)

# This will concurrently run while the jobs are being processed.
for i in range(0, 100):
    print("Waitting for results...")
    time.sleep(0.5)
    if end:
        break

print("Running jobs synchronously...")
results = qp.run_batch(qobjs)
for result in results:
    print("result: {}".format(result))
    try:
        print(result.get_counts("rippleadd"))
    except Exception as ex:
        print("ERROR: {}".format(ex))

    print("============")

print("Done")
