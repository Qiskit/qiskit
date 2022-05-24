

from vqe_icans import *



from qiskit.circuit.library import EfficientSU2
from qiskit import Aer, transpile, assemble


ansatz = EfficientSU2(3)
parameters = list(ansatz.parameters)


backend = Aer.get_backend("qasm_simulator")

hamilt0 = PauliSumOp.from_list(
    [
        ("XXI", 1),
        ("XIX", 1),
        ("IXX", 1),
        ("YYI", 1),
        ("YIY", 1),
        ("IYY", 1),
        ("ZZI", 1),
        ("ZIZ", 1),
        ("IZZ", 1),
        ("IIZ", 3),
        ("IZI", 3),
        ("ZII", 3),
    ]
)

#Parameters Icans
max_iterations = 5
min_shots = 100
alpha = 0.05
L = 2
mu = 0.99
b =0.000001

icans1 = ICANS(ansatz, quantum_instance=backend, minShots = min_shots, alpha = alpha, max_iterations= max_iterations,  limit_learning_rate = True)

result_Icans = icans1.compute_minimum_eigenvalue(operator = hamilt0)
print(result_Icans)