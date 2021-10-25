import numpy as np
import scipy as sp
import os
import time
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['QISKIT_IN_PARALLEL'] = 'True'


from scipy.integrate import Radau, ode, solve_ivp, RK45, RK23
from qiskit.working_files.varQTE.implicit_euler import BDF, backward_euler_fsolve

from qiskit.quantum_info import partial_trace


from qiskit import Aer, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import EfficientSU2, RealAmplitudes
from qiskit.circuit import ParameterVector
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.library import RYGate

from qiskit.opflow.evolutions.varqtes.varqrte import VarQRTE
from qiskit.opflow.evolutions.varqtes.varqite import VarQITE
from qiskit.opflow.evolutions.varqte import ForwardEuler

from qiskit.opflow import StateFn, SummedOp, MatrixOp, TensoredOp, OperatorBase
from qiskit.opflow import Z, I, Y, X

from qiskit.working_files.QMRF.MRF_Hamiltonian_generator import MrfHamiltonianGenerator
from qiskit.working_files.QMRF.ansatz_generator import AnsatzGenerator
np.random.seed = 11

# Evolution time
t = 0.1

num_time_steps = 10
depths = [1]


ode_solvers = [RK45]
ode_solvers_names = ['RK45']
regs = [None]
reg_names = ['lstsq']


C = [[0, 1, 2], [3, 4, 5], [2, 3]]  # clique structure
n = 6 # number nodes/qubits
mode = 'fast'
observable = MrfHamiltonianGenerator().gen_Hamiltonian(C, n, mode)
if not mode == 'fast_pauli':
    observable = MatrixOp(observable).to_pauli_op()
observable = observable.reduce()
target = sp.linalg.expm(-observable.to_matrix(massive=True))/np.trace(sp.linalg.expm(
    -observable.to_matrix(massive=True)))
print('Target ', np.diag(target))


#
# params3 = ParameterVector('p3', 12)
# ansatz3 = QuantumCircuit(3)
# ansatz3.h(0)
# ansatz3.h(1)
# ansatz3.h(2)
# ansatz3.ry(params3[0], 0)
# ansatz3.ry(params3[1], 1)
# ansatz3.ry(params3[2], 2)
# ansatz3.cx(0, 1)
# ansatz3.cx(0, 2)
# ansatz3.ry(params3[3], 0)
# ansatz3.ry(params3[4], 1)
# ansatz3.ry(params3[5], 2)
# ansatz3.cx(1, 2)
# ansatz3.cx(1, 0)
# ansatz3.ry(params3[6], 0)
# ansatz3.ry(params3[7], 1)
# ansatz3.ry(params3[8], 2)
# ansatz3.cx(2, 0)
# ansatz3.cx(2, 1)
# # ansatz3.cry(params3[3], 0, 1)
# # ansatz3.cry(params3[4], 1, 2)
# # ansatz3.cry(params3[5], 2, 0)
# ansatz3.ry(params3[9], 0)
# ansatz3.ry(params3[10], 1)
# ansatz3.ry(params3[11], 2)
#
# print(ansatz3)
#
# params4 = ParameterVector('p4', 12)
# ansatz4 = QuantumCircuit(3)
# ansatz4.h(0)
# ansatz4.h(1)
# ansatz4.h(2)
# ansatz4.ry(params4[0], 0)
# ansatz4.ry(params4[1], 1)
# ansatz4.ry(params4[2], 2)
# ansatz4.cx(0, 1)
# ansatz4.ry(params4[3], 0)
# ansatz4.ry(params4[4], 1)
# ansatz4.cx(1, 0)
# ansatz4.cx(1, 2)
# ansatz4.ry(params4[5], 1)
# ansatz4.ry(params4[6], 2)
# ansatz4.cx(2, 1)
# ansatz4.cx(2, 0)
# ansatz4.ry(params4[7], 0)
# ansatz4.ry(params4[8], 2)
# ansatz4.cx(0, 2)
# ansatz4.ry(params4[9], 0)
# ansatz4.ry(params4[10], 1)
# ansatz4.ry(params4[11], 2)
#
# print(ansatz4)
# ansatz4.cry(params4[3], 0, 1)
# ansatz4.cry(params4[4], 1, 0)
# ansatz4.cry(params4[5], 1, 2)
# ansatz4.cry(params4[6], 2, 1)
# ansatz4.cry(params4[7], 2, 0)
# ansatz4.cry(params4[8], 0, 2)


# ansaetze = [ansatz1, ansatz2, ansatz3, ansatz4]
# params = [params1, params2, params3, params4]

# ansaetze = [ansatz3, ansatz4]
# params = [params3, params4]
ansatz, params = AnsatzGenerator().get_ansatz0(C, n)
ansaetze = [ansatz]
params = [params]
# for nts in num_time_steps:
# nts = num_time_steps[1]
for l, ansatz in enumerate(ansaetze):
    for k, ode_solver in enumerate(ode_solvers):
        for d in depths:
            for j, reg in enumerate(regs):
                print(ode_solvers_names[k])
                print(reg_names[j])
                parameters = params[l]

                init_param_values = np.zeros(len(parameters))
                # Now we stack the observable and the quantum state together.
                # The evolution time needs to be added as a coefficient to the operator
                op = ~StateFn(observable) @ StateFn(ansatz)
                op = t * op

                print('---------------------------------------------------------------------')
                print(ansatz)
                t0 = time.time()
                varqite_snapshot_dir = os.path.join('..', 'xor_withoutwork_bigger', 'imag',
                                                    'ansatz'+str(l),
                                                    reg_names[j],
                                                    ode_solvers_names[k] + 'nat_grad')

                varqite = VarQITE(parameters=parameters, grad_method='lin_comb',
                                  init_parameter_values=init_param_values,
                                  num_time_steps=num_time_steps,
                                  ode_solver=ode_solver,
                                  backend=Aer.get_backend('statevector_simulator'),
                                  regularization=reg,
                                  error_based_ode=False,
                                  snapshot_dir=varqite_snapshot_dir)
                approx_time_evolved_state_imag = varqite.convert(op)
                out_state = approx_time_evolved_state_imag.eval().primitive.data
                out_state = np.multiply(out_state, np.conj(out_state))

                print('Output', out_state)
                varqite_error_bounds = varqite.error_bound(
                    varqite_snapshot_dir, imag_reverse_bound=False, H=observable.to_matrix(
                        massive=True))
                np.save(os.path.join(varqite_snapshot_dir, 'error_bounds.npy'),
                        varqite_error_bounds)
                # dir_fast = '../output/imag/10/ridge/RK45error'
                # varqite.print_results([dir_fast], [os.path.join(dir_fast,
                #                                                'error_bounds.npy')])
                varqite.plot_results([varqite_snapshot_dir], [os.path.join(varqite_snapshot_dir,
                                                              'error_bounds.npy')]
                                      )

                print('run time', (time.time()-t0)/60)
