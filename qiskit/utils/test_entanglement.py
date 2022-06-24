from qiskit.utils.circuit_utils import entanglement
from qiskit.utils.entanglement.parametric_circuits import ansatz

import numpy as np
from qiskit import Aer


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
            ent_cap = entanglement(ansatze, backend, num_params=10).von_neumann_entanglement_qutip()
            entanglement_cap[i, layers] = (ent_cap)
        # entanglement_cap = np.array(entanglement_cap)
    file_name = "entropy_run"+str(k)
    np.savetxt(file_name + ".csv", entanglement_cap, delimiter = "\t")

