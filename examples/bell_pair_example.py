from qiskit import QuantumCircuit
from qiskit_aer.primitives import Sampler

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

sampler = Sampler()
result = sampler.run([qc]).result()

# Convert quasi-distributions to counts
shots = 1000
counts = {format(k, '0{}b'.format(qc.num_qubits)): int(v*shots)
          for k, v in result.quasi_dists[0].items()}

print("Bell state counts:", counts)
