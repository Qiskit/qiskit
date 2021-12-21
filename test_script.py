from qiskit.circuit.library.standard_gates.i import IGate
from qiskit.circuit.library.standard_gates.z import ZGate
from qiskit.circuit.library.standard_gates.y import YGate
from qiskit.circuit.library.standard_gates.x import XGate
from qiskit.algorithms.optimizers.slsqp import SLSQP
from qiskit.algorithms import QAOA
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.algorithms import AdaptQAOA

import numpy as np
from qiskit.algorithms.optimizers import NELDER_MEAD, COBYLA
from max_cut import max_cut_hamiltonian
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
params = {'legend.fontsize': 'xx-large',
        'figure.figsize': (12, 12),
        'axes.labelsize': 'xx-large',
        'axes.titlesize':'xx-large',
        'xtick.labelsize':'x-large',
        'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)
import math

from qiskit.algorithms.minimum_eigen_solvers.adapt_qaoa import adapt_mixer_pool
from functools import reduce
from qiskit import BasicAer
from qiskit import QuantumRegister, QuantumCircuit
from itertools import combinations_with_replacement, permutations, product
from qiskit.circuit.library import IGate, XGate, YGate, ZGate
def _string_to_qiskit(qstring):
    from qiskit.opflow import I, PauliSumOp, X, Y, Z
    qis_dict = {"I": I, "X": X, "Y": Y, "Z": Z}

    if all(x == qstring[0] for x in qstring):
        gate = qstring[0]
        list_string = [i * "I" + gate + (len(qstring) - i - 1) * "I" for i in range(len(qstring))]

        return sum(
            [
                reduce(lambda a, b: a ^ b, [qis_dict[char.upper()] for char in x])
                for x in list_string
            ]
        )
    return reduce(lambda a, b: a ^ b, [qis_dict[char.upper()] for char in qstring])


def _create_mixer_pool(num_q, add_multi, circ):
    """Compute the mixer pool
    Args:
        num_q (int): number of qubits
        add_multi (bool): whether to add multi qubit gates to the mixer pool
        circ (bool): if output mixer pool in form of list of circuits instead of list of operators
        parameterize (bool): if the circuit mixers should be parameterized

    Returns:
        list: all possible combinations of mixers
    """
    mixer_pool = ["X" * num_q]

    mixer_pool.append("Y" * num_q)

    mixer_pool += [i * "I" + "X" + (num_q - i - 1) * "I" for i in range(num_q)]
    mixer_pool += [i * "I" + "Y" + (num_q - i - 1) * "I" for i in range(num_q)]

    if add_multi:
        indicies = list(permutations(range(num_q), 2))
        indicies = list(set(tuple(sorted(x)) for x in indicies))
        combos = list(combinations_with_replacement(["X", "Y", "Z"], 2))

        full_multi = list(product(indicies, combos))
        for item in full_multi:
            iden_str = list("I" * num_q)
            iden_str[item[0][0]] = item[1][0]
            iden_str[item[0][1]] = item[1][1]

            mixer_pool.append("".join(iden_str))

    mixer_circ_list = []
    for mix_str in mixer_pool:
        if circ:
            qr = QuantumRegister(num_q)
            qc = QuantumCircuit(qr)
            for i, mix in enumerate(mix_str):
                qiskit_dict = {"I": IGate(), "X": XGate(), "Y": YGate(), "Z": ZGate()}

                mix_qis_gate = qiskit_dict[mix]
                qc.append(mix_qis_gate, [i])
                mixer_circ_list.append(qc)
        else:
            op = _string_to_qiskit(mix_str)
            mixer_circ_list.append(op)
    return mixer_circ_list


STATEVECTOR_SIMULATOR = QuantumInstance(
    BasicAer.get_backend("statevector_simulator"),
    seed_simulator=10598,
    seed_transpiler=10598,
)
W1 = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]])
P1 = 1
M1 = adapt_mixer_pool(num_qubits=2, pool_type='multi')
S1 = {"0101", "1010"}

W2 = np.array(
    [
        [0.0, 8.0, -9.0, 0.0],
        [8.0, 0.0, 7.0, 9.0],
        [-9.0, 7.0, 0.0, -8.0],
        [0.0, 9.0, -8.0, 0.0],
    ]
)
P2 = 1
M2 = None
S2 = {"1011", "0100"}

def extend_initial_points(max_reps, gamma_0 = 0.01, beta_0 = -np.pi/4):
    return [beta_0]*max_reps+[gamma_0]*max_reps

def _sample_most_likely(state_vector):
    """Compute the most likely binary string from state vector.
    Args:
        state_vector (numpy.ndarray or dict): state vector or counts.

    Returns:
        numpy.ndarray: binary string as numpy.ndarray of ints.
    """
    n = int(np.log2(state_vector.shape[0]))
    k = np.argmax(np.abs(state_vector))
    x = np.zeros(n)
    for i in range(n):
        x[i] = k % 2
        k >>= 1
    return x

def _get_operator(weight_matrix):
    from qiskit.opflow import PauliSumOp
    from qiskit.quantum_info import Pauli
    """Generate Hamiltonian for the max-cut problem of a graph.

    Args:
        weight_matrix (numpy.ndarray) : adjacency matrix.

    Returns:
        PauliSumOp: operator for the Hamiltonian
        float: a constant shift for the obj function.

    """
    num_nodes = weight_matrix.shape[0]
    pauli_list = []
    shift = 0
    for i in range(num_nodes):
        for j in range(i):
            if weight_matrix[i, j] != 0:
                x_p = np.zeros(num_nodes, dtype=bool)
                z_p = np.zeros(num_nodes, dtype=bool)
                z_p[i] = True
                z_p[j] = True
                pauli_list.append([0.5 * weight_matrix[i, j], Pauli((z_p, x_p))])
                shift -= 0.5 * weight_matrix[i, j]
    opflow_list = [(pauli[1].to_label(), pauli[0]) for pauli in pauli_list]
    return PauliSumOp.from_list(opflow_list), shift

def _get_graph_solution(x: np.ndarray) -> str:
    """Get graph solution from binary string.

    Args:
        x : binary string as numpy array.

    Returns:
        a graph solution as string.
    """

    return "".join([str(int(i)) for i in 1 - x])

def test_adapt_qaoa_qc_mixer(w, prob, solutions, convert_to_matrix_op):
    optimizer = optimizer = COBYLA(maxiter=1000000, tol=0)
    qubit_op, _ = _get_operator(w)
    if convert_to_matrix_op:
        qubit_op = qubit_op.to_matrix_op()

    num_qubits = qubit_op.num_qubits
    mixer = _create_mixer_pool(num_q=num_qubits, add_multi=True, circ=True)

    adapt_qaoa = AdaptQAOA(
        optimizer=optimizer,
        max_reps=prob,
        mixer_pool=mixer,
        quantum_instance=STATEVECTOR_SIMULATOR,
    )

    result = adapt_qaoa.compute_minimum_eigenvalue(operator=qubit_op, disp=True)
    x = _sample_most_likely(result.eigenstate)
    graph_solution = _get_graph_solution(x)
    print("Problem solutions {}".format(solutions))
    print("Computed result {}".format(graph_solution))
    print("Test result: {}".format(graph_solution in solutions))

# test_inputs = [[W2, P2, S2, False],[W2, P2, S2, True]]

# test_result = test_adapt_qaoa_qc_mixer(W1, P1, S1, False)
# test_result = test_adapt_qaoa_qc_mixer(W2, P2, S2, False)


max_reps = 12
D, nq = 5, 6
quantum_instance = QuantumInstance(Aer.get_backend('qasm_simulator'), shots=1064) 
optimiser = COBYLA(maxiter=1000000, tol=0)#maxfev=100000, xatol=0.00005)#, adaptive=True)#maxiter=(1+max_reps) * 3000, adaptive=True, xatol=0.00002, tol=0.00002)
cost_op = max_cut_hamiltonian(D=D, nq=nq)
gs_energy = min(np.real(np.linalg.eig(cost_op.to_matrix())[0]))
init_pt = extend_initial_points(max_reps=max_reps)

# adaptqaoa = AdaptQAOA(quantum_instance=quantum_instance,mixer_pool_type='multi', max_reps=max_reps,
#                         optimizer=optimiser, initial_point=init_pt[:2*max_reps])
# final_result, total_results = adaptqaoa.compute_minimum_eigenvalue(cost_op, iter_results = True, disp=True)
"--------------------------------------------------------------"
"run adapt"
"--------------------------------------------------------------"
import copy
print(f"Problem ground state energy: {gs_energy}")
adapt_vals_dict = {'multi':0, 'single':0}#, 'singular':0}
adapt_val_dict = copy.copy(adapt_vals_dict)
for mt in adapt_vals_dict.keys():
    print("Running adapt with mixer pool type {}".format(mt))
    adaptqaoa = AdaptQAOA(max_reps=max_reps, quantum_instance=quantum_instance,mixer_pool_type=mt, 
                            optimizer=optimiser, threshold=0,initial_point=init_pt)
    final_result, total_results = adaptqaoa.compute_minimum_eigenvalue(cost_op, iter_results = True, disp = True)
    adapt_depth = len(total_results)
    adapt_vals_dict[mt] = [(total_results[i].optimal_value-gs_energy) for i in range(adapt_depth)]

# "--------------------------------------------------------------"
# "now run regular qaoa over the maximum number of iterations!!!!"
# "--------------------------------------------------------------"
qaoa_vals = []
for p in range(1,adapt_depth+1):
    print("Depth: {}".format(p))
    kappa = init_pt[:max_reps][:p]
    ip = init_pt[:max_reps][:p]+init_pt[max_reps:][:p]
    qaoa = QAOA(reps=p, quantum_instance=quantum_instance, optimizer=optimiser, initial_point=ip)
    out = qaoa.compute_minimum_eigenvalue(cost_op)
    qaoa_vals.append(out.optimal_value-gs_energy)
    print("Initial point: {}".format(ip))
    print("Optimal value: {}".format(out.optimal_value))
    print("Optimal parameters: {}".format(out.optimal_parameters))
    print("Relative energy: {}".format(qaoa_vals[-1]))
# import time
# adapt_depth = 10
# igamma = [0.01,0.01+np.pi, 0.01+10*np.pi, 0.01+100*np.pi]
# ibeta = [-np.pi/4,-np.pi/4-np.pi,-np.pi/4-10*np.pi,-np.pi/4-100*np.pi]
# fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
# cm = ['r','b','orange','green']
# for i in range(4):
#     qaoa_vals = []
#     delta_beta, delta_gamma, time_p = [], [], []
#     init_pt = extend_initial_points(max_reps=max_reps, gamma_0=igamma[i], beta_0=ibeta[i])
#     for p in range(1,adapt_depth+1):
#         print("Depth: {}".format(p))
#         start = time.time()
#         kappa = init_pt[:max_reps][:p]
#         ip = init_pt[:max_reps][:p]+init_pt[max_reps:][:p]
#         qaoa = QAOA(reps=p, quantum_instance=quantum_instance, optimizer=optimiser, initial_point=ip)
#         out = qaoa.compute_minimum_eigenvalue(cost_op)
#         diff_op_ip = np.abs(np.array(list(dict(zip(qaoa.ansatz.parameters,ip)).values()))-np.array(list(out.optimal_parameters.values())))
#         delta_beta.append(np.mean(np.split(diff_op_ip, 2)[0])%np.pi)
#         delta_gamma.append(np.mean(np.split(diff_op_ip, 2)[1])%np.pi)
#         qaoa_vals.append(out.optimal_value-gs_energy)
#         time_p.append(time.time()-start)
#         print("Initial point: {}".format(ip))
#         print("Optimal value: {}".format(out.optimal_value))
#         print("Optimal parameters: {}".format(out.optimal_parameters))
#         print("Relative energy: {}".format(qaoa_vals[-1]))
#     beta_label = r'$β_0$'
#     gamma_label = r'$γ_0$'
#     next_diff = r'$+0\cdot\pi$'
#     if i==1:
#         next_diff = r'+$\pi$'
#         beta_label += next_diff
#         gamma_label += next_diff
#     if i>1:
#         next_diff = r'+{}$\pi$'.format([10,100][i-2])
#         beta_label+= next_diff
#         gamma_label+= next_diff
#     ax[0].plot(np.arange(1, adapt_depth+1), delta_beta, '-', color=cm[i], label=beta_label)
#     ax[0].plot(np.arange(1, adapt_depth+1), delta_gamma, '--', color=cm[i], label=gamma_label)
#     ax[0].legend()
#     ax[1].plot(np.arange(1,adapt_depth+1), qaoa_vals, cm[i], label=next_diff)
#     ax[1].legend()
#     print('------------------------------------------------------------------')
# ax[0].set_title(r"Initial values: [β, γ] = [{}, {}]".format(ibeta[0],igamma[0]))
# ax[1].set_ylabel("Energy")
# ax[0].set_ylabel("Mean Change in Hyperparameter Magnitude")
# ax[1].set_xlabel("Depth (p)")
# plt.savefig('qaoa_buggin_pi_adam.png')

# ax[2].plot(np.arange(1, adapt_depth+1), time_p, label='β')
# ax[2].set_xlabel("Depth")
# ax[2].set_ylabel("Computation time (s)")
# plt.savefig('qaoa_buggin_pi.png')



plt.plot(np.arange(1,adapt_depth+1),np.log10(qaoa_vals+1e-16),label='QAOA')
for mt in adapt_vals_dict.keys():
    plt.plot(np.arange(1,adapt_depth+1),np.log10(adapt_vals_dict[mt]+1e-16),label="ADAPT-"+mt)
plt.xlabel("Circuit depth")
plt.ylabel(r"$\log_{10}(\Delta E)$")
plt.legend()
plt.savefig('adaptqaoa_vs_qaoa.png')