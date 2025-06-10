from qiskit import QuantumCircuit, qpy
import time

# Describe bell state circuit:
qc_bell = QuantumCircuit(2,name="Bell")
qc_bell.h(0)
qc_bell.cx(0,1)
qc_bell.measure_all()

# Describe equal state circuit:
qc_equal = QuantumCircuit(1, name="Equal")
qc_equal.h(0)
qc_equal.measure_all()

if __name__ == "__main__":
    # Store 5 & 5 into a QPY file:
    total_time = 0
    for i in range(1000):
        with open("x_my_circuit.qpy", "wb") as fd:
            start_time = time.time()
            qpy.dump(([qc_bell]*5)+([qc_equal]*5), fd)
            end_time = time.time()
            elapsed_time = end_time - start_time
            total_time += elapsed_time
    print(f"Average dump time: {total_time/1000:.6f} seconds")

    # Retrieve it:
    total_time = 0
    for i in range(1000):
        retrieved = None
        with open("x_my_circuit.qpy", "rb") as fd:
            start_time = time.time()
            retrieved = qpy.load(fd)
            end_time = time.time()
            elapsed_time = end_time - start_time
            total_time += elapsed_time
    print(f"Average load time (parallel): {total_time/1000:.6f} seconds")

'''
Average dump time: 0.001076 seconds
Average load time (non-parallel): 0.001300 seconds
'''

'''
Average dump time: 0.001041 seconds
Average load time (parallel): 0.002287 seconds
'''

'''
Average dump time: 0.001070 seconds
Average load time (parallel with buffering): 0.001676 seconds
'''

'''
PROBLEMS: Buffering is not scalable and is still slower (maybe due to memory swapping when populating the RAM)
Also, Python file objects ar not thread-safe, so parallel loading is not possible.
Also, Python has  a GIL...
'''

'''
Average dump time: 0.001034 seconds
Average load time (producer-consumer approach): 0.001576 seconds
'''
    

# with open("x_my_circuit.qpy", "wb") as fd:
#     qpy.dump(([qc_bell]*5)+([qc_equal]*5), fd)


# retrieved = None
# with open("x_my_circuit.qpy", "rb") as fd:
#     retrieved = qpy.load(fd)
    
# # Checking if they are correct:
# for qc in retrieved:
#     print(qc.draw())
#     print('*'*20)