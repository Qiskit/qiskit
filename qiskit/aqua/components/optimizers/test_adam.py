import numpy as np
from qiskit.aqua.components.optimizers import Optimizer
from qiskit.aqua.components.optimizers.adam_amsgrad import Adam

#Test Function to be Optimized
def test(x):
    return [x[0]**2+x[1]**2]

#Set Optimizer
optimizer = Adam(amsgrad=True)
number_variables = 2
#Evaluate Optimization
x_fin, test_fin, eval_fin = optimizer.optimize(number_variables, test)
#Print
print(x_fin)
print(test_fin)
print(eval_fin)