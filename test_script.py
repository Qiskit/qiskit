from qiskit.algorithms import QAOA
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.algorithms import AdaptQAOA

import numpy as np
from qiskit.algorithms.optimizers import NELDER_MEAD
from max_cut import max_cut_hamiltonian

def extend_initial_points(max_reps, gamma_0 = 0.01, beta_0 = np.pi/4):
    return [beta_0]*max_reps+[gamma_0]*max_reps

D, nq = 3, 4
cost_op = max_cut_hamiltonian(D=D, nq=nq)
gs_energy = min(np.real(np.linalg.eig(cost_op.to_matrix())[0]))
quantum_instance = QuantumInstance(Aer.get_backend('qasm_simulator'), shots=1024)
optimiser = NELDER_MEAD()
max_reps = 8
# adaptqaoa = AdaptQAOA(max_reps=max_reps, quantum_instance=quantum_instance,mixer_pool=[X^I^I^I + ^X^I^I + X^I^I^I ],
#                         optimizer=optimiser, threshold=0)#,
"--------------------------------------------------------------"
"run adapt"
"--------------------------------------------------------------"
import copy
adapt_vals_dict = {'Multi':0, 'Single':0, 'Singular':0}
adapt_val_dict = copy.copy(adapt_vals_dict)
for mt in adapt_vals_dict.keys():
    print("Running adapt with mixer pool type {}".format(mt))
    adaptqaoa = AdaptQAOA(max_reps=max_reps, quantum_instance=quantum_instance,mixer_pool_type=mt, optimizer=optimiser, threshold=0,
                            initial_point=extend_initial_points(max_reps=max_reps))
    final_result, total_results = adaptqaoa.compute_minimum_eigenvalue(cost_op, iter_results = True)
    adapt_depth = len(total_results)
    adapt_vals_dict[mt] = [(total_results[i].optimal_value-gs_energy) for i in range(adapt_depth)]
    adapt_val_dict[mt] = adaptqaoa.initial_point

# "--------------------------------------------------------------"
# "now run regular qaoa over the maximum number of iterations!!!!"
# "--------------------------------------------------------------"
# print("Beginning QAOA experiment up to circuit depth {}".format(adapt_depth))
# ad_ip = adapt_val_dict[min(adapt_vals_dict)]
# init_beta = ad_ip[:max_reps+1]
# init_gamma = ad_ip[max_reps+1:]
adapt_depth=2
qaoa_vals = []
for p in range(1,adapt_depth+1):
    qaoa = QAOA(reps=p, quantum_instance=quantum_instance, optimizer=optimiser)#, initial_point=np.concatenate((init_beta[:p],init_gamma[:p])))
    out = qaoa.compute_minimum_eigenvalue(cost_op)
    qaoa_vals.append(out.optimal_value-gs_energy)
    print("Depth {}:".format(p))
    print("Optimal value: {}".format(out.optimal_value))
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