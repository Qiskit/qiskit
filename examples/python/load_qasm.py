"""
Example on how to use: load_qasm_file

"""
from qiskit.wrapper import load_qasm_file
from qiskit import QISKitError, execute, Aer

try:
    qc = load_qasm_file("examples/qasm/entangled_registers.qasm")

    # See a list of available local simulators
    print("Aer backends: ", Aer.backends())
    sim_backend = Aer.get_backend('qasm_simulator')


    # Compile and run the Quantum circuit on a local simulator backend
    job_sim = execute(qc, sim_backend)
    sim_result = job_sim.result()

    # Show the results
    print("simulation: ", sim_result)
    print(sim_result.get_counts(qc))

except QISKitError as ex:
    print('There was an internal Qiskit error. Error = {}'.format(ex))

