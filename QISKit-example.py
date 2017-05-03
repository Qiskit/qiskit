"""
QISKit funtionalities example

Author: Ismael Faro
"""

from qiskit import QuantumProgram
import json


with open('token.json') as data_file:
    token = json.load(data_file)["token"]
print(token)

# Define Program Specifications.
QPSpecs = {
    "name": "algorithm",
    "circuits": [{
        "name": "demo",
        "quantum_registers": {
            "name":"q",
            "size":3
        },
        "classical_registers": {
            "name":"c",
            "size":3
        }}],

}

#  Define a program using the previus specifications
QP_program = QuantumProgram(specs=QPSpecs)

# Get the program componentes, like Circuits and Registers
# qc, qr, cr = QP_program.quantum_elements()

# It's possible get individual componentes
qc = QP_program.circuit("demo")
qr = QP_program.quantum_registers("q")
cr = QP_program.classical_registers('c')

'''
Test Gates
'''
qc.u3(0.3, 0.2, 0.1, qr[0])
qc.h(qr[1])
qc.cx(qr[1], qr[2])
qc.barrier()

qc.cx(qr[0], qr[1])
qc.h(qr[0])
qc.measure(qr[0], cr[0])
qc.measure(qr[1], cr[1])

qc.z(qr[2]).c_if(cr, 1)
qc.x(qr[2]).c_if(cr, 1)
qc.measure(qr[2], cr[2])

'''
test Circuit functionalities
'''
# obtain QASM
print(qc.qasm())


'''
test Program functionalities
'''
# create a new circuit
qr2 = QP_program.create_quantum_registers("qr2", 10)
cr2 = QP_program.create_classical_registers("cr2", 5)
qc2 = QP_program.create_circuit("circuit2", "qr2", "cr2")

print(qc2.qasm())

QP_program.set_api_token(token)
print(QP_program.api_config())

device = 'qx5q' # the device to run on
shots = 1024    #the number of shots in the experiment. 

id = QP_program.run_circuit("demo", device, shots, max_credits=3)
print('run Circuit', id)



ids = QP_program.run_program( device, shots, max_credits=3)
print('run Program', ids)

# TODO: Topology definition
# topology={
#     hardware={},
#     map={}
# }

# TODO: 
# sim1 = myQP.set_scope(topology=topology)
# topology2={
#     map={}
# }


# sim2 = myQP.set_scope( topology=topology2)

# sim1.compile.execute.plot()
# sim2.compile.execute.plot()


# sim1 = myQP.set_scope(hardware={}, map={}, topology={})

# myQP.compile()
#   myQP.parse(versionQasm, qfiles)
#   myQP.unroller()
#   myQP.optimizer(standar)
#   myQP.map(topology, operations)
#   myQP.optimizer(cleaner)
# myQP.execute()

# myQP.execute()
# myQP.execute(debug = {})


# myQP.plot()

# hardware.status()
# hardware.command()