"""
Example on how to use: load_qasm_file

Note: if you have only cloned the Qiskit repository but not
used `pip install`, the examples only work from the root directory.
"""
from qiskit.wrapper import load_qasm_file
from qiskit import QISKitError, available_backends, execute
try:
    qc = load_qasm_file("examples/qasm/entangled_registers.qasm")

    # See a list of available local simulators
    print("Local backends: ", available_backends({'local': True}))

    # Compile and run the Quantum circuit on a local simulator backend
    job_sim = execute(qc, "local_qasm_simulator")
    sim_result = job_sim.result()

    # Show the results
    print("simulation: ", sim_result)
    print(sim_result.get_counts(qc))

except QISKitError as ex:
    print('There was an internal Qiskit error. Error = {}'.format(ex))

