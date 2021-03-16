"""Utils for execution."""

import numpy as np
from qiskit import Aer
from qiskit.opflow import StateFn, Gradient, NaturalGradient, AerPauliExpectation, CircuitSampler


def get_objective(observable, ansatz, parameters, sampler=None):
    """Get a handle for an objective function to compute the expected value, given parameters."""
    # if statement outside of function call for performance
    if sampler is None:
        def objective_function(x):
            bound = ansatz.bind_parameters(dict(zip(parameters, x)))
            expr = ~StateFn(observable) @ bound
            return expr.eval().real

    elif sampler == 'aerpauli':
        sampler = CircuitSampler(Aer.get_backend('qasm_simulator'))
        expectation = AerPauliExpectation()

        def objective_function(x):
            bound = ansatz.bind_parameters(dict(zip(parameters, x)))
            expr = ~StateFn(observable) @ bound
            expr = expectation.convert(expr)
            res = sampler.convert(expr).eval().real
            return res

    else:
        def objective_function(x):
            bound = ansatz.bind_parameters(dict(zip(parameters, x)))
            expr = ~StateFn(observable) @ bound
            res = sampler.convert(expr).eval().real
            return res

    return objective_function


def get_overlap(ansatz, parameters, sampler=None):
    """Get the Fubini-Study metric for this application."""

    if sampler is None:
        def overlap(x, y):
            bound_x = ansatz.bind_parameters(dict(zip(parameters, x)))
            bound_y = ansatz.bind_parameters(dict(zip(parameters, y)))
            overlap = (~bound_x @ bound_y).eval()
            return -0.5 * np.abs(overlap).real ** 2

    elif sampler == 'aerpauli':
        sampler = CircuitSampler(Aer.get_backend('qasm_simulator'))
        expectation = AerPauliExpectation()

        def overlap(x, y):
            bound_x = ansatz.bind_parameters(dict(zip(parameters, x)))
            bound_y = ansatz.bind_parameters(dict(zip(parameters, y)))
            expr = expectation.convert(~bound_x @ bound_y)
            overlap = sampler.convert(expr).eval()
            return -0.5 * np.abs(overlap).real ** 2

    else:
        def overlap(x, y):
            bound_x = ansatz.bind_parameters(dict(zip(parameters, x)))
            bound_y = ansatz.bind_parameters(dict(zip(parameters, y)))
            overlap = sampler.convert(~bound_x @ bound_y).eval()
            return -0.5 * np.abs(overlap).real ** 2

    return overlap


def get_gradient(gradient_base, observable, ansatz, parameters, sampler=None):
    """Get a handle for a gradient function of the expected value, given parameters."""
    # if statement outside of function call for performance
    if sampler is None:
        def gradient_function(x):
            expr = ~StateFn(observable) @ ansatz
            grad = gradient_base.convert(expr, params=parameters)
            bound = grad.bind_parameters(dict(zip(parameters, x)))
            return np.real(bound.eval())

    elif sampler == 'aerpauli':
        sampler = CircuitSampler(Aer.get_backend('qasm_simulator'))
        expectation = AerPauliExpectation()

        def gradient_function(x):
            expr = ~StateFn(observable) @ ansatz
            grad = gradient_base.convert(expr, params=parameters)
            bound = grad.bind_parameters(dict(zip(parameters, x)))
            bound = expectation.convert(bound)
            res = np.real(sampler.convert(bound).eval())
            return res

    else:
        def gradient_function(x):
            expr = ~StateFn(observable) @ ansatz
            grad = gradient_base.convert(expr, params=parameters)
            bound = grad.bind_parameters(dict(zip(parameters, x)))
            res = np.real(sampler.convert(bound).eval())
            return res

    return gradient_function


def get_vanilla_gradient(observable, ansatz, parameters, sampler=None):
    """Get the vanilla gradient."""
    return get_gradient(Gradient(), observable, ansatz, parameters, sampler)


def get_natural_gradient(observable, ansatz, parameters, sampler=None):
    """Get the vanilla gradient."""
    return get_gradient(NaturalGradient(), observable, ansatz, parameters, sampler)


def get_finite_difference_gradient(eps, observable, ansatz, parameters, sampler=None):
    """Get the finite difference gradient."""
    objective = get_objective(observable, ansatz, parameters, sampler)

    dim = len(parameters)

    def finite_difference(x):
        gradient = np.empty(dim)
        for i in range(dim):
            e_i = np.identity(dim)[:, i]
            gradient[i] = (objective(x + eps * e_i) - objective(x - eps * e_i)) / (2 * eps)
        return gradient

    return finite_difference
