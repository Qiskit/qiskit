# useful additional packages
import sys, getpass
import matplotlib.pyplot as plt
import numpy as np

# importing the QISKit
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, QISKitError, QuantumJob
from qiskit import available_backends, execute, register, get_backend, compile

try:
    sys.path.append("../../")  # go to parent dir
    import Qconfig

    qx_config = {
        "APItoken": Qconfig.APItoken,
        "url": Qconfig.config['url']}
    print('Qconfig loaded from %s.' % Qconfig.__file__)
except:
    APItoken = getpass.getpass('Please input your token and hit enter: ')
    qx_config = {
        "APItoken": APItoken,
        "url": "https://quantumexperience.ng.bluemix.net/api"}
    print('Qconfig.py not found in qiskit-tutorial directory; Qconfig loaded using user input.')

# import basic plot tools
from qiskit.tools.visualization import plot_histogram

n = 15 # the length of the first register for querying the oracle

# Choose a type of oracle at random. With probability half it is constant,
# and with the same probability it is balanced
oracleType, oracleValue = np.random.randint(2), np.random.randint(2)

if oracleType == 0:
    print("The oracle returns a constant value ", oracleValue)
else:
    print("The oracle returns a balanced function")
    a = np.random.randint(1, 2 ** n)  # this is a hidden parameter for balanced oracle.

# Creating registers
# n qubits for querying the oracle and one qubit for storing the answer
qr = QuantumRegister(n + 1)  # all qubits are initialized to zero
# for recording the measurement on the first register
cr = ClassicalRegister(n)

circuitName = "DeutschJosza"
djCircuit = QuantumCircuit(qr, cr)

# Create the superposition of all input queries in the first register by applying the Hadamard gate to each qubit.
for i in range(n):
    djCircuit.h(qr[i])

# Flip the second register and apply the Hadamard gate.
djCircuit.x(qr[n])
djCircuit.h(qr[n])

# Apply barrier to mark the beginning of the oracle
djCircuit.barrier()

if oracleType == 0:  # If the oracleType is "0", the oracle returns oracleValue for all input.
    if oracleValue == 1:
        djCircuit.x(qr[n])
    else:
        djCircuit.iden(qr[n])
else:  # Otherwise, it returns the inner product of the input with a (non-zero bitstring)
    for i in range(n):
        if (a & (1 << i)):
            djCircuit.cx(qr[i], qr[n])

# Apply barrier to mark the end of the oracle
djCircuit.barrier()

# Apply Hadamard gates after querying the oracle
for i in range(n):
    djCircuit.h(qr[i])

# Measurement
for i in range(n):
    djCircuit.measure(qr[i], cr[i])



#draw the circuit
from qiskit.tools.visualization import circuit_drawer
circuit_drawer(djCircuit)



#register token to connect with remote backends
register(qx_config['APItoken'], qx_config['url']) #must register to connect with remote backends
print("Available backends:", available_backends())

backend = "ibmq_qasm_simulator"
shots = 1000
job = execute(djCircuit, backend=backend, shots=shots)
results = job.result()
answer = results.get_counts()

plot_histogram(answer)

import time  # for sleep

backend = "ibmq_qasm_simulator"
# backend = "ibmqx5" #uncomment this to run on real device ibmqx5
shots = 1000
if get_backend(backend).status['available'] == True:

    job = execute(djCircuit, backend=backend, shots=shots)
    lapse = 0
    interval = 10
    while not job.done:
        print('Status @ {} seconds'.format(interval * lapse))
        print(job.status)
        time.sleep(interval)
        lapse += 1
    print(job.status)

    results = job.result()
    answer = results.get_counts()

    threshold = int(0.03 * shots)  # the threshold of plotting significant measurements
    filteredAnswer = {k: v for k, v in answer.items() if v >= threshold}  # filter the answer for better view of plots

    removedCounts = np.sum([v for k, v in answer.items() if v < threshold])  # number of counts removed
    filteredAnswer['other_bitstring'] = removedCounts  # the removed counts are assigned to a new index

    plot_histogram(filteredAnswer)

    print(filteredAnswer)
