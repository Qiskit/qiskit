import numpy as np

from qiskit.aqua.operators import SummedOp, StateFn, Z, I, ListOp
from qiskit.aqua.operators.gradients import NaturalGradient
from qiskit.circuit.library import EfficientSU2

# Evolution time
t =  10
num_qubits = 4
# Instantiate the model ansatz
depth = 1
entangler_map = [[i+1, i] for i in range(num_qubits - 1)]
ansatz = EfficientSU2(num_qubits, reps=depth, entanglement = entangler_map)

def combo_fn(x):
    # log-likelihood
    amplitudes = x[0].primitive.data
    pdf = np.multiply(amplitudes, np.conj(amplitudes))
    return np.sum(np.log(pdf))/(-len(amplitudes))

def grad_combo_fn(x):
    amplitudes = x[0].primitive.data
    pdf = np.multiply(amplitudes, np.conj(amplitudes))
    grad = []
    for p in pdf:
        grad += [-1/p]
    return grad

# qc = RealAmplitudes(2, reps=2)
# lambda x: jnp.sum(jnp.log(x))/(-len(x))
op = ListOp([StateFn(ansatz)], combo_fn=combo_fn, grad_combo_fn=grad_combo_fn)
# grad = Gradient().convert(grad_op, qc.ordered_parameters)

# Set the discretization grid of the time steps
num_time_steps = 10
time_steps = np.linspace(0, t, num_time_steps)

# Convert the operator that holds the Hamiltonian and Ansatz using the NaturalGradient
nat_grad = NaturalGradient(grad_method='lin_comb', regularization='ridge'
                           ).convert(op, ansatz.ordered_parameters)

# Initialize the Ansatz parameters
param_values = np.random.rand(len(ansatz.ordered_parameters))

# Propagate the Ansatz parameters step by step (here with explicit Euler)
for step in time_steps:
    param_dict = dict(zip(ansatz.ordered_parameters, param_values))
    nat_grad_result = np.real(nat_grad.assign_parameters(param_dict).eval())
    param_values = list(np.subtract(param_values, t/num_time_steps * np.real(
        nat_grad_result.flatten())))
    print('Loss',
          op.assign_parameters(dict(zip(ansatz.ordered_parameters, param_values))).eval())

print('Final values', param_values)
