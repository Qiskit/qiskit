import numpy as np
from qiskit.opflow import ListOp, StateFn, X, Y, Z, One
x, y, z = ~One @ One, 2 * (~One @ One), 3 * (~One @ One)
def blowup(triu):
    print(triu)
    dim = len(triu)
    matrix = np.empty((dim, dim))
    for i in range(dim):
        for j in range(dim - i):
            matrix[i, i + j] = triu[i][j]
            if j != 0:
                matrix[i + j, i] = triu[i][j]
    return matrix
triu = ListOp([
    ListOp([x, x, x]),
    ListOp([y, y]),
    ListOp([z]),
],
    combo_fn=blowup
)
print(triu.eval())