# from libraries.meyer_wallach_measure import compute_Q_ptrace, compute_Q_ptrace_qiskit


from qiskit.providers import backend
from qiskit.utils.circuit_utils import entanglement

# from inflection import parameterize
# from matplotlib import backend_bases
# from qiskit.utils.

from qiskit.utils.entanglement.parametric_circuits import ansatz
# from qiskit import Aer, execute, transpile
# import qiskit.quantum_info as qi 
import numpy as np
from qiskit import Aer
# import statistics

total_circuit = 1
feature_dim = 4
repitition = 1
num_eval = 1
entanglement_cap = np.zeros((total_circuit,repitition))
backend = Aer.get_backend("statevector_simulator")

for k in range(num_eval):
        

    for i in range(total_circuit):
        circuit_id = i+1

        for layers in range(repitition):

            ansatze  = ansatz(layers+1, feature_dim, circuit_id).get_ansatz()
            ent_cap = entanglement(ansatze, backend, num_params=10).get_entanglement()
            entanglement_cap[i, layers] = (ent_cap)
        # entanglement_cap = np.array(entanglement_cap)
    file_name = "vn_entropy_run"+str(k)
    np.savetxt(file_name + ".csv", entanglement_cap, delimiter = "\t")

