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
def extend_initial_points(max_reps, gamma_0 = 0.01, beta_0 = -np.pi/4):
    return [beta_0]*max_reps+[gamma_0]*max_reps
    # return [beta_0-100*np.pi]*max_reps+[gamma_0+100*np.pi]*max_reps

def rand_ip(max_reps):
    from qiskit.utils import algorithm_globals
    return list(algorithm_globals.random.uniform(2*max_reps*[-2000 * np.pi], 2*max_reps*[2000 * np.pi]))
max_reps = 1
D, nq = 5, 6
quantum_instance = QuantumInstance(Aer.get_backend('qasm_simulator'), shots=1064) 
optimiser = COBYLA(maxiter=1000000, tol=0)#maxfev=100000, xatol=0.00005)#, adaptive=True)#maxiter=(1+max_reps) * 3000, adaptive=True, xatol=0.00002, tol=0.00002)
cost_op = max_cut_hamiltonian(D=D, nq=nq)
gs_energy = min(np.real(np.linalg.eig(cost_op.to_matrix())[0]))
init_pt = extend_initial_points(max_reps=max_reps)

adaptqaoa = AdaptQAOA(quantum_instance=quantum_instance,mixer_pool_type='multi', max_reps=max_reps,
                        optimizer=optimiser, initial_point=init_pt[:2*max_reps])
final_result, total_results = adaptqaoa.compute_minimum_eigenvalue(cost_op, iter_results = True, disp=True)
final_result, total_results = adaptqaoa.compute_minimum_eigenvalue(cost_op.to_matrix_op(), iter_results = True, disp=True)
"--------------------------------------------------------------"
"run adapt"
"--------------------------------------------------------------"
# import copy
# print(f"Problem ground state energy: {gs_energy}")
# adapt_vals_dict = {'multi':0, 'single':0}#, 'singular':0}
# adapt_val_dict = copy.copy(adapt_vals_dict)
# for mt in adapt_vals_dict.keys():
#     print("Running adapt with mixer pool type {}".format(mt))
#     adaptqaoa = AdaptQAOA(max_reps=max_reps, quantum_instance=quantum_instance,mixer_pool_type=mt, 
#                             optimizer=optimiser, threshold=0,initial_point=init_pt)
#     final_result, total_results = adaptqaoa.compute_minimum_eigenvalue(cost_op, iter_results = True)
#     adapt_depth = len(total_results)
#     adapt_vals_dict[mt] = [(total_results[i].optimal_value-gs_energy) for i in range(adapt_depth)]

# "--------------------------------------------------------------"
# "now run regular qaoa over the maximum number of iterations!!!!"
# "--------------------------------------------------------------"
# qaoa_vals = []
# for p in range(1,adapt_depth+1):
#     print("Depth: {}".format(p))
#     kappa = init_pt[:max_reps][:p]
#     ip = init_pt[:max_reps][:p]+init_pt[max_reps:][:p]
#     qaoa = QAOA(reps=p, quantum_instance=quantum_instance, optimizer=optimiser, initial_point=ip)
#     out = qaoa.compute_minimum_eigenvalue(cost_op)
#     qaoa_vals.append(out.optimal_value-gs_energy)
#     print("Initial point: {}".format(ip))
#     print("Optimal value: {}".format(out.optimal_value))
#     print("Optimal parameters: {}".format(out.optimal_parameters))
#     print("Relative energy: {}".format(qaoa_vals[-1]))
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



# plt.plot(np.arange(1,adapt_depth+1),np.log10(qaoa_vals),label='QAOA')
# for mt in adapt_vals_dict.keys():
#     plt.plot(np.arange(1,adapt_depth+1),np.log10(adapt_vals_dict[mt]),label="ADAPT-"+mt)
# plt.xlabel("Circuit depth")
# plt.ylabel(r"$\log_{10}(\Delta E)$")
# plt.legend()
# plt.savefig('adaptqaoa_vs_qaoa.png')