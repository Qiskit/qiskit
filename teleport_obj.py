"""
Quantum teleportation example based on OPENQASM example.

Author: Andrew Cross
"""
from qiskit import QuantumProgram

# define Program
# add circuit to Program
# add registers to circuit (or program?)

# TODO: Create Scope object to pass Hardware, Map and others.

# QPScope = {
#     topology={
#             hardware={},
#             map={}
#     },

# }

QPSpecs = {
    "name": "algorithm",
    # "importmodule": "",
    "circuits": [{
        "name": "demo",
        #"importqasm": "chemisty.qjson",
        "quantum_registers": {
            "name":"q",
            "size":3
        },
        "classical_registers": {
            "name":"c",
            "size":3
        }}]
}

QP_program = QuantumProgram(specs=QPSpecs)
# c = QP_program.set_circuits("circuit name"")
# q = QP_program.quantum_registers['q']


#cr = QP_program.get_classical_registers('c')

qc, qr, cr = QP_program.quantum_elements()
print(qc)

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

print(qc.qasm())


# use methods instead - or have method as well
# c1 = a + b + c
# c2 = a + bp + c

# chemistry1 = make_variational_state + do_measurement_1
# chemistry2 = make_variational_state + do_measurement_2

# p.add_circuit(c1)
