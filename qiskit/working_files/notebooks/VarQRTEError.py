#!/usr/bin/env python
# coding: utf-8

# ## VarQRTE Error

# In[6]:


from jax import grad, jit
import jax.numpy as jnp

import os
import warnings

import numpy as np
import scipy as sp
from scipy.linalg import expm

from qiskit.circuit.library import EfficientSU2, RealAmplitudes

from qiskit.opflow import StateFn, MatrixOp, CircuitOp, X, I, PauliOp, Z, Y, SummedOp
from qiskit.opflow.gradients import NaturalGradient
from qiskit.opflow.evolutions.varqtes.varqrte import VarQRTE

np.random.seed = 2


# In[7]:


# Set Hamiltonian
# Hamiltonian = SummedOp([(Z ^ X), 0.3 *(Y ^ Y), (Y ^ I)]).reduce()
Hamiltonian = SummedOp([(Z ^ X), 0.8 *(Y ^ Y)]).reduce() # works
# Hamiltonian = SummedOp([(Z ^ X), (X ^ Z), 3 * (Z ^ Z)]).reduce()
# Hamiltonian = observable = SummedOp([0.25 * (I ^ X), 0.25 * (X ^ I), (Z ^ Z)]).reduce()
# Hamiltonian = SummedOp([(Z ^ X), 3. * (Y ^ Y), (Z ^ X), (I ^ Z), (Z ^ I)]).reduce() #works
# Hamiltonian = (Z ^ X) # works
# Hamiltonian = (Y ^ I)
# Set time and time_steps 
time = 1
time_steps = 1


# In[8]:


T = Hamiltonian **2
print(T.to_pauli_op().reduce())


# In[9]:


# Helper Function computing the inner product
def inner_prod(x, y):
    return np.matmul(np.conj(np.transpose(x)), y)


# In[10]:


# Set Ansatz and initial parameters
ansatz = EfficientSU2(2, reps=1, entanglement='linear')
parameters = ansatz.ordered_parameters
# init_param_values = np.random.rand(len(ansatz.ordered_parameters))
# init_param_values = np.zeros(len(ansatz.ordered_parameters))
# for i in range(ansatz.num_qubits):
#     init_param_values[-(ansatz.num_qubits + i + 1)] = np.pi / 2
    
# print('init param values ', init_param_values)
init_param_values = [0.78641215, 0.38487082, 0.76062407, 0.42770714, 0.26507494, 0.58129185,
                    0.05290203, 0.70437158]

# ### Analytic Calculations

# In[11]:


def analytic_error(state, H, et_ket=None):
    if et_ket is None:
        et_ket = np.zeros(len(state))
        
    dt_state = -1j*(np.dot(H, state) + et_ket)

    # <dt|dt>
    et = inner_prod(dt_state, dt_state) 
    #<H^2>
    et += inner_prod(state, np.matmul(np.matmul(H, H), state))
    # 2Im<dt|H|state>
    et -= 2 * np.imag(inner_prod(dt_state, np.matmul(H, state)))
    
    print('Gradient error', np.round(np.sqrt(et), 5))
    return np.sqrt(et), dt_state


# In[12]:


num_qubits = Hamiltonian.num_qubits
H = Hamiltonian.to_matrix(massive=True)

# Propagation Unitary
evolution_op = lambda t: expm(-1 * H * t)
# Initial State
init_state = StateFn(ansatz).assign_parameters(dict(zip(parameters, init_param_values))).eval().primitive

evolution_op = lambda t: expm(-1j * H * t)
target_state = lambda t: np.dot(evolution_op(t), init_state)
dt_target_state = lambda t: np.dot(-1j * H, target_state(t))


error = 0
prepared_state = target_state(0)
for j in range(0, time_steps):
    et_ket = np.ones(2 ** num_qubits) / np.sqrt(2 ** num_qubits) * j * 1e-4

    et, dt_prepared_state = analytic_error(prepared_state, H, et_ket=et_ket)
    # Euler State propagation
    prepared_state += time/time_steps * dt_prepared_state
    # Compute Error
#     error += time/time_steps * et
    error += time/time_steps * np.sqrt(np.linalg.norm(et_ket))

print('error bound for time', np.round(time/time_steps*(j+1), 3), np.round(error, 3))

# sqrt of the l2-norm of the distance between the perturbed, prepared state and the exat, target state
print('True error ', np.sqrt(np.linalg.norm(prepared_state - target_state(time))))


# ### Numpy Calculations

# In[13]:


def dt_params(a, c, regularization=None):
    a = np.real(a)
    c = np.real(c)
    if regularization:
        # If a regularization method is chosen then use a regularized solver to
        # construct the natural gradient.
        nat_grad = NaturalGradient._regularized_sle_solver(
            a, c, regularization=regularization)
    else:
        nat_grad = np.linalg.lstsq(a, c, rcond=None)[0]
        if np.linalg.norm(nat_grad) < 1e-8:
            nat_grad = NaturalGradient._regularized_sle_solver(a,
                                                               c,
                                                               regularization='perturb_diag')
            warnings.warn(r'Norm of the natural gradient smaller than $1e^{-8}$ use '
                          r' `perturb_diag` regularization.')
        if np.linalg.norm(nat_grad) > 1e-4:
            nat_grad = NaturalGradient._regularized_sle_solver(a,
                                                               c,
                                                               regularization='ridge')
            warnings.warn(r'Norm of the natural gradient bigger than $1e^{3}$ use '
                          r' `ridge` regularization.')
#         try:
#             # Try to solve the system of linear equations Ax = C.
#             nat_grad = np.linalg.solve(a, c)
#         except np.linalg.LinAlgError:  # singular matrix
#             print('Singular matrix lstsq solver required')
#             nat_grad, resids, _, _ = np.linalg.lstsq(a, c)
#             print('Residuals', resids)
    return np.real(nat_grad)


# In[14]:


def numpy_error(a, c, dt_weights, H, state):

    dtdt = inner_prod(dt_weights, np.matmul(a, dt_weights))
    et = dtdt
    print('dtdt', dtdt)
    
    h_squared = inner_prod(state, np.matmul(np.matmul(H, H), state))
    
    if h_squared < dtdt:
        print('Eq. 8 violated')
    et = np.add(et, h_squared)
    
    print('H^2', h_squared)

    
    dt = 2*inner_prod(dt_weights, c)
    et -= dt
    print('2Im<dt|H|>', dt)
    
    print('Grad error', np.round(np.sqrt(et), 6))
    print('|adtw-c|', np.linalg.norm(np.matmul(a, dt_weights)-c))
    return np.sqrt(et)


# In[15]:


print('Warning: Numpy Calculations are only compatible with SU2 depth 1 - else hard-coded changes needed.')

def ry(theta):
    return jnp.array([[jnp.cos(theta/2.), -1*jnp.sin(theta/2)], [jnp.sin(theta/2),
                                                                 jnp.cos(theta/2)]])

def rz(theta):
    return jnp.array([[jnp.exp(-1j * theta / 2.), 0], [0, jnp.exp(1j * theta / 2.)]])


def ryry(alpha, beta):
    return jnp.kron(ry(alpha), ry(beta))

def rzrz(alpha, beta):
    return jnp.kron(rz(alpha), rz(beta))

cx = jnp.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
i = jnp.eye(2)
y = jnp.array([[0, -1j], [1j, 0]])
z = jnp.array([[1, 0], [0, -1]])
iy = -0.5j * jnp.kron(i, y)
yi = -0.5j * jnp.kron(y, i)
iz = -0.5j * jnp.kron(i, z)
zi = -0.5j * jnp.kron(z, i)

init = jnp.array([1, 0, 0, 0])

def state_fn(params):
    vec = jnp.dot(jnp.dot(rzrz(params[7], params[6]), ryry(params[5], params[4])),
                  jnp.dot(cx, jnp.dot(rzrz(params[3],  params[2]), jnp.dot(ryry(params[1],
                                                                                params[0]),
                                                                           init))))
    return vec

def state0(params):
    vec = jnp.dot(jnp.dot(rzrz(params[7], params[6]), ryry(params[5], params[4])),
                  jnp.dot(cx, jnp.dot(rzrz(params[3],  params[2]), jnp.dot(ryry(params[1],
                                                                                params[0]),
                                                                           init))))
    return vec[0]

def state1(params):
    vec = jnp.dot(jnp.dot(rzrz(params[7], params[6]), ryry(params[5], params[4])),
                  jnp.dot(cx, jnp.dot(rzrz(params[3],  params[2]), jnp.dot(ryry(params[1],
                                                                                params[0]),
                                                                           init))))
    return vec[1]

def state2(params):
    vec = jnp.dot(jnp.dot(rzrz(params[7], params[6]), ryry(params[5], params[4])),
                  jnp.dot(cx, jnp.dot(rzrz(params[3],  params[2]), jnp.dot(ryry(params[1],
                                                                                params[0]),
                                                                           init))))
    return vec[2]

def state3(params):
    vec = jnp.dot(jnp.dot(rzrz(params[7], params[6]), ryry(params[5], params[4])),
                  jnp.dot(cx, jnp.dot(rzrz(params[3],  params[2]), jnp.dot(ryry(params[1],
                                                                                params[0]),
                                                                           init))))
    return vec[3]


def A(vec, gradient):
    vec = np.reshape(vec, (len(vec), 1))
    a = np.real((inner_prod(gradient, gradient)))
    print('a carrying part', np.round(a, 3))
    a = np.subtract(a, np.real(np.matmul(inner_prod(gradient, vec),
                    (inner_prod(vec, gradient)))))
    print('a phase fix', np.round(np.real(np.matmul(inner_prod(gradient, vec),
                    np.transpose(np.conj(inner_prod(gradient, vec))))), 3))
    return a

def C(vec, gradient, h):
    vec = np.reshape(vec, (len(vec), 1))
    c = np.imag(inner_prod(gradient, np.matmul(h, vec)))
    print('carrying part c', np.round(c, 3))
    c = np.add(c, np.real(1j * inner_prod(gradient, vec) * inner_prod(vec, np.matmul(h, vec))))
    print('phase fix c', np.round(np.real(1j * inner_prod(gradient, vec) * inner_prod(vec, np.matmul(h, vec))), 3))
#     print('c', c)
    return c

def grad0(params):
    try:
        return grad(state0)(params)
    except Exception:
        return grad(state0, holomorphic=True)(jnp.complex64(params))
def grad1(params):
    try:
        return grad(state1)(params)
    except Exception:
        return grad(state1, holomorphic=True)(jnp.complex64(params))
def grad2(params):
    try:
        return grad(state2)(params)
    except Exception:
        return grad(state2, holomorphic=True)(jnp.complex64(params))
def grad3(params):
    try:
        return grad(state3)(params)
    except Exception:
        return grad(state3, holomorphic=True)(jnp.complex64(params))


# In[16]:


init_param_values = np.zeros(len(ansatz.ordered_parameters))
for i in range(ansatz.num_qubits):
    init_param_values[-(ansatz.num_qubits + i + 1)] = np.pi / 2
init_param_values = [0.78641215, 0.38487082, 0.76062407, 0.42770714, 0.26507494, 0.58129185,
                         0.05290203, 0.70437158]
params = init_param_values
error = 0

for j in range(time_steps):

    # dim vec x num_params
    gradient = [grad0(params), grad1(params), grad2(params), grad3(params)]
    gradient = np.array([[complex(item) for item in g] for g in gradient]).astype(
        np.complex)

    state = state_fn(params)
    state = np.array([complex(s) for s in state]).astype(np.complex)

    metric = A(state, gradient)
    c_grad = C(state, gradient, H)
    print('grad res', np.round(c_grad*2, 3))
    print('metric', np.round(metric, 3))

    dt_weights = dt_params(metric, c_grad, 'ridge')
    print('dt_weights', dt_weights)

    et = numpy_error(metric, c_grad, dt_weights, H, state)

    error += time/time_steps * et

    params += time/time_steps * np.reshape(dt_weights, np.shape(params))
    print('params', params)
    print('error bound for time', np.round(time/time_steps*(j+1), 3), np.round(error, 3))

# sqrt of the l2-norm of the distance between the perturbed, prepared state and the exat, target state
print('True error ', np.sqrt(np.linalg.norm(state_fn(params) - target_state(time))))
print(state_fn(params))
print(target_state(time))


# ### Variational Calculations

# In[17]:


import os
from qiskit import Aer
init_param_values = np.zeros(len(ansatz.ordered_parameters))
for i in range(ansatz.num_qubits):
    init_param_values[-(ansatz.num_qubits + i + 1)] = np.pi / 2
init_param_values = [0.78641215, 0.38487082, 0.76062407, 0.42770714, 0.26507494, 0.58129185,
                    0.05290203, 0.70437158]
op = ~StateFn(Hamiltonian)@StateFn(ansatz)
from qiskit.utils import QuantumInstance
backend = QuantumInstance(Aer.get_backend('statevector_simulator'), shots=100000)
# op = ~StateFn(ansatz)@Hamiltonian@StateFn(ansatz)
op = time * op
approx_time_evolved_state = VarQRTE(parameters=parameters,
                                    grad_method='lin_comb',
                                    init_parameter_values=init_param_values,
                                    num_time_steps=time_steps,
                                    regularization='ridge', backend=backend,
                                    error_based_ode=False,
                                    snapshot_dir=os.path.join('dummy')).convert(op)


# In[18]:


test = ~StateFn(ansatz)@Hamiltonian@StateFn(ansatz)
print(test.assign_parameters(dict(zip(ansatz.ordered_parameters, init_param_values))).eval())


# In[19]:


print(test)


# In[20]:


Hamiltonian


# In[ ]:




