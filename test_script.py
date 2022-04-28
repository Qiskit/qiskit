from functools import reduce
from qiskit.algorithms.optimizers.optimizer import Optimizer
from qiskit.opflow import I, X, Y, Z
from qiskit.algorithms import QAOA
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.algorithms import AdaptQAOA

import networkx as nx
import numpy as np
from qiskit.algorithms.optimizers import NELDER_MEAD
import random
from qiskit.opflow.primitive_ops import MatrixOp

QISKIT_DICT = {"I": I, "X": X, "Y": Y, "Z": Z}


def build_maxcut_hamiltonian(G, return_pauli_strings=False):
    import copy
    from qiskit.quantum_info.operators import Pauli
    num_qubits = G.order()
    iden_str = list("I" * num_qubits)
    H_terms = []
    H_str_terms = []
    iden = (I^(num_qubits)).to_matrix()
    H_str_terms.append((iden_str, G.size()))
    for edge in list(G.edges):
        term = copy.deepcopy(iden_str)
        term[edge[0]] = "Z"
        term[edge[1]] = "Z"
        pauli_str = ''
        for i in term:
            pauli_str += i
        term_mat = Pauli(pauli_str).to_matrix()
        weight = G.edges[edge[0], edge[1]]['weight']
        H_str_terms.append((term, weight))
        H_terms.append(-0.5*weight*(iden - term_mat))

    if return_pauli_strings:
        return H_str_terms
    else:
        return sum(H_terms)


def string_to_qiskit(qstring):
    if is_all_same(qstring):
        # case where its all X's or Y's
        gate = qstring[0]
        list_string = [
            i * "I" + gate + (len(qstring) - i - 1) * "I" for i in range(len(qstring))]
        return sum([reduce(lambda a, b: a ^ b, [QISKIT_DICT[char.upper()] for char in x]) for x in list_string])

    return reduce(lambda a, b: a ^ b, [QISKIT_DICT[char.upper()] for char in qstring])
def is_all_same(items):
    return all(x == items[0] for x in items)

D, nq = 3, 6
G = nx.random_regular_graph(D, nq, seed=1234) # connectivity, vertices
for (u, v) in G.edges():
    G.edges[u,v]['weight'] = random.randint(0,1000)/1000
cost_op = MatrixOp(build_maxcut_hamiltonian(G)).to_pauli_op()



# mixer_list = ["XXIII","XIIX","IXXII"]
# cost_op = string_to_qiskit("ZZZZZ")
# cost_op = string_to_qiskit("IIIZZ")# + string_to_qiskit("ZZIII")
# mixer_list = [string_to_qiskit(x) for x in mixer_list]

quantum_instance = QuantumInstance(Aer.get_backend('qasm_simulator'), shots=1024)

'''
adapt = AdaptQAOA(mixer_pool_type='singular', max_reps=5, quantum_instance=quantum_instance)
#adapt.optimal_mixer_list = mixer_list
cme = adapt.compute_minimum_eigenvalue(cost_op)
print(cme)
'''
max_reps = 3
adaptqaoa = AdaptQAOA(max_reps=max_reps, quantum_instance=quantum_instance, mixer_pool_type="multi", optimizer=NELDER_MEAD(),
                        initial_point=[np.pi/4, 0.01]*max_reps)
out = adaptqaoa.run_adapt(cost_op)
# print("Adapt result: ", out)
print("Optimal cost",adaptqaoa.get_optimal_cost())
# print(adaptqaoa.get_optimal_circuit().draw())

# qaoa = QAOA(reps=5, quantum_instance=quantum_instance)
# out = qaoa.compute_minimum_eigenvalue(cost_op)
# print(out)
# print(qaoa.get_optimal_circuit().draw())
