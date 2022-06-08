from qiskit.algorithms.optimizers import GradientDescent
import numpy as np

def learning_rate():
                power = 0.6
                constant_coeff = 0.1

                def powerlaw():
                    n = 0
                    while True:
                        yield constant_coeff * (n ** power)
                        n += 1

                return powerlaw()

def f(x):
    return (np.linalg.norm(x) - 1) ** 2

def grad_f(x):
    return 2 * (np.linalg.norm(x) - 1) * x / np.linalg.norm(x)

initial_point = np.array([1, 0.5, -0.2])

optimizer = GradientDescent(maxiter=20, learning_rate=learning_rate)

result = optimizer.minimize(x0 = initial_point , fun = f , jac = grad_f)

print(result)

def learning_rate():
            power = 0.6
            constant_coeff = 0.1

            def powerlaw():
                n = 0
                while True:
                    yield constant_coeff * (n**power)
                    n += 1

            return powerlaw()

print(next(learning_rate()))
