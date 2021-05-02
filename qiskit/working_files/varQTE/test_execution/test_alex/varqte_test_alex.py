import numpy as np
import time
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['QISKIT_IN_PARALLEL'] = 'True'

from scipy.integrate import RK45

from qiskit import Aer

from qiskit.circuit import ParameterVector
from qiskit.opflow.evolutions.varqtes.varqrte import VarQRTE
from qiskit.opflow.evolutions.varqte import ForwardEuler

from qiskit.opflow import StateFn, MatrixOp
np.random.seed = 11

from qiskit.working_files.varQTE.test_execution.test_alex import SpinBosonHamiltonian, SBHamil
H = SpinBosonHamiltonian([1, 1], 2, omega=1, g_couple=0.5, delta=0, epsilon=-1,
                         coupling='x')
H_matrix = H.as_matrix()
H_op = MatrixOp(H_matrix).to_pauli_op()
varform = SBHamil([1, 1], 2, coupling='x', trotter_depth=[1, 1])
x = ParameterVector('x', varform.num_parameters)
init_params = np.zeros(varform.num_parameters)
varform_circ = varform.construct_circuit(x)

# Evolution time
t = 1
op = StateFn(H_op, is_measurement=True) @ StateFn(varform_circ)
op = t * op

num_time_steps = [10]
depths = [1]

ode_solvers = [ForwardEuler, RK45]
ode_solvers_names = ['ForwardEuler', 'RK45']
regs = ['ridge', 'perturb_diag', None]
reg_names = ['ridge', 'perturb_diag', 'lstsq']

dir = 'test_alex_output'

for nts in num_time_steps:
    for k, ode_solver in enumerate(ode_solvers):
        for d in depths:
            for j, reg in enumerate(regs):
                print(ode_solvers_names[k])
                print(reg_names[j])
                varqrte_snapshot_dir = os.path.join(dir, 'real',
                                                    reg_names[j],
                                                    ode_solvers_names[k] + 'nat_grad')
                t0 = time.time()
                try:
                    varqrte = VarQRTE(parameters=x[:],
                                    grad_method='lin_comb',
                                    init_parameter_values=init_params,
                                    num_time_steps=nts,
                                    ode_solver=ode_solver,
                                    backend=Aer.get_backend('statevector_simulator'),
                                    regularization=reg,
                                    error_based_ode=False,
                                    snapshot_dir=varqrte_snapshot_dir
                                    )
                    approx_time_evolved_state_real = varqrte.convert(op)
                    varqrte_error_bounds = varqrte.error_bound(varqrte_snapshot_dir)
                    np.save(os.path.join(varqrte_snapshot_dir, 'error_bounds.npy'),
                            varqrte_error_bounds)
                    #
                    print('run time', (time.time()-t0)/60)
                    varqrte.plot_results([varqrte_snapshot_dir],
                                         [os.path.join(varqrte_snapshot_dir,
                                                       'error_bounds.npy')])
                except Exception:
                    pass
