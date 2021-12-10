from qiskit.algorithms.optimizers.slsqp import SLSQP
from qiskit.algorithms import QAOA
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.algorithms import AdaptQAOA

import numpy as np
from qiskit.algorithms.optimizers import NELDER_MEAD, SLSQP
from max_cut import max_cut_hamiltonian

def extend_initial_points(max_reps, gamma_0 = 0.01, beta_0 = -np.pi/4):
    # return [beta_0-np.pi]*max_reps+[gamma_0+np.pi]*max_reps
    return [beta_0-np.pi]*max_reps+[gamma_0+np.pi]*max_reps

def rand_ip(max_reps):
    from qiskit.utils import algorithm_globals
    return list(algorithm_globals.random.uniform(2*max_reps*[-2000 * np.pi], 2*max_reps*[2000 * np.pi]))
max_reps = 16
D, nq = 3, 6
cost_op = max_cut_hamiltonian(D=D, nq=nq)
gs_energy = min(np.real(np.linalg.eig(cost_op.to_matrix())[0]))
init_pt = extend_initial_points(max_reps=max_reps)
quantum_instance = QuantumInstance(Aer.get_backend('qasm_simulator'), shots=1064) 
optimiser = NELDER_MEAD(disp=True)#, tol=1e-08)#, adaptive=True)#maxiter=(1+max_reps) * 3000, adaptive=True, xatol=0.00002, tol=0.00002)
# optimiser = SLSQP(maxiter= (1+max_reps) * 1000, ftol=1e-08)

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
    final_result, total_results = adaptqaoa.compute_minimum_eigenvalue(cost_op, iter_results = True)
    adapt_depth = len(total_results)
    adapt_vals_dict[mt] = [(total_results[i].optimal_value-gs_energy) for i in range(adapt_depth)]

"--------------------------------------------------------------"
"now run regular qaoa over the maximum number of iterations!!!!"
"--------------------------------------------------------------"
qaoa_vals = []
for p in range(1,adapt_depth+1):
    kappa = init_pt[:max_reps][:p]
    ip = init_pt[:max_reps][:p]+init_pt[max_reps:][:p]
    qaoa = QAOA(reps=p, quantum_instance=quantum_instance, optimizer=optimiser, initial_point=ip)
    out = qaoa.compute_minimum_eigenvalue(cost_op)
    qaoa_vals.append(out.optimal_value-gs_energy)
    print("Depth: {}".format(p))
    print("Initial point: {}".format(ip))
    print("Optimal value: {}".format(out.optimal_value))
    print("Optimal parameters: {}".format(out.optimal_parameters))
    print("Relative energy: {}".format(qaoa_vals[-1]))

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
params = {'legend.fontsize': 'xx-large',
          'figure.figsize': (12, 12),
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)
plt.plot(np.arange(1,adapt_depth+1),qaoa_vals,label='QAOA')
for mt in adapt_vals_dict.keys():
    plt.plot(np.arange(1,adapt_depth+1),adapt_vals_dict[mt],label="ADAPT-"+mt)
plt.xlabel("Circuit depth")
plt.ylabel("Energy")
plt.legend()
plt.savefig('adaptqaoa_vs_qaoa.png')