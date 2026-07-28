import time

from qiskit.circuit import ClassicalRegister, QuantumCircuit

start = time.time()
size = 10
qc = QuantumCircuit(*(ClassicalRegister(5, f"cr_{i:05}") for i in range(size)))
end = time.time()
elapsed = (end - start) * 1000
print("n=10: ", elapsed)

start = time.time()
size = 100
qc = QuantumCircuit(*(ClassicalRegister(5, f"cr_{i:05}") for i in range(size)))
end = time.time()
elapsed = (end - start) * 100
print("n=100: ", elapsed)

start = time.time()
size = 1000
qc = QuantumCircuit(*(ClassicalRegister(5, f"cr_{i:05}") for i in range(size)))
end = time.time()
elapsed = (end - start) * 1000
print("n=1000: ", elapsed)

start = time.time()
size = 10_000
qc = QuantumCircuit(*(ClassicalRegister(5, f"cr_{i:05}") for i in range(size)))
end = time.time()
elapsed = (end - start) * 1000
print("n=10000: ", elapsed)

qc = QuantumCircuit(ClassicalRegister(5, "a"), ClassicalRegister(5, "b"))
