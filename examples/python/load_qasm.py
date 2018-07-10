"""
Example on how to use: load_qasm_file
If you want to use your local cloned repository intead of the one installed via pypi,
you have to run like this:
    examples/python$ PYTHONPATH=$PYTHONPATH:../.. python load_qasm.py
"""
from qiskit.wrapper import load_qasm_file
from qiskit import QISKitError, available_backends, execute
try:
    qc = load_qasm_file("../qasm/entangled_registers.qasm")

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

