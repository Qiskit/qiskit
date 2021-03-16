import numpy as np
# import jax.numpy as jnp
from qiskit.circuit.library import RealAmplitudes
from qiskit.aqua.operators import StateFn, Gradient, ListOp


def combo_fn(x):
    # return 0
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




qc = RealAmplitudes(2, reps=2)
# lambda x: jnp.sum(jnp.log(x))/(-len(x))
grad_op = ListOp([StateFn(qc)], combo_fn=combo_fn, grad_combo_fn=grad_combo_fn)
grad = Gradient().convert(grad_op, qc.ordered_parameters)

param_dict = dict(zip(qc.ordered_parameters, np.random.rand(len(qc.ordered_parameters))))
print(grad.assign_parameters(param_dict).eval())