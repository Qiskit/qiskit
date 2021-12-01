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
from max_cut import max_cut_hamiltonian

def extend_initial_points(max_reps, gamma_0 = 0.01, beta_0 = np.pi/4):
    return [beta_0]*max_reps+[gamma_0]*max_reps

D, nq = 3, 6
cost_op = max_cut_hamiltonian(D=D, nq=nq)
gs_energy = min(np.real(np.linalg.eig(cost_op.to_matrix())[0]))
quantum_instance = QuantumInstance(Aer.get_backend('qasm_simulator'), shots=1024)
optimiser = NELDER_MEAD()
max_reps = 4
# run adapt
adaptqaoa = AdaptQAOA(max_reps=max_reps, quantum_instance=quantum_instance, mixer_pool_type="multi", optimizer=optimiser)#
                        # initial_point=extend_initial_points(max_reps=max_reps))
final_result, total_results = adaptqaoa.run_adapt(cost_op)
adapt_depth = len(total_results)
print("Optimal values for adapt:")
print([total_results[i].optimal_value for i in range(adapt_depth)])
adapt_vals = [(total_results[i].optimal_value-gs_energy) for i in range(adapt_depth)]

# now run regular qaoa over the maximum number of iterations!!!!
print("Beginning QAOA experiment up to circuit depth {}".format(adapt_depth))
qaoa_vals = []
for p in range(1,adapt_depth+1):
    qaoa = QAOA(reps=p, quantum_instance=quantum_instance, optimizer=optimiser)
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
plt.plot(np.arange(1,adapt_depth+1),adapt_vals,label='AdaptQAOA')
plt.xlabel("Circuit depth")
plt.ylabel("Energy")
plt.legend()
plt.savefig('adaptqaoa_vs_qaoa.png')



