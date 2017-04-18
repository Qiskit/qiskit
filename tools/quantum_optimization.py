"""
Quantum Optimization tools.

These are simple methods for common tasks in our optimization,

Author: Jay Gambetta
"""


def cost_classical(data, n, alpha, beta):
    """Compute the cost function.

    n = number of qubits
    alpha is a vector with elements q0 -- qn
    beta is a matrix of couplings
    """
    temp = 0
    tot = sum(data.values())
    for key in data:
        observable = 0
        for j in range(len(key)):
            if key[j] == '0':
                observable = observable + alpha[n-1-j]
            elif key[j] == '1':
                observable = observable - alpha[n-1-j]
            for i in range(j):
                if key[j] == '0' and key[i] == '0':
                    observable = observable + beta[n-1-i, n-1-j]
                elif key[j] == '1' and key[i] == '1':
                    observable = observable + beta[n-1-i, n-1-j]
                elif key[j] == '0' and key[i] == '1':
                    observable = observable - beta[n-1-i, n-1-j]
                elif key[j] == '1' and key[i] == '0':
                    observable = observable - beta[n-1-i, n-1-j]
            for i in range(j+1, n):
                if key[j] == '0' and key[i] == '0':
                    observable = observable + beta[n-1-i, n-1-j]
                elif key[j] == '1' and key[i] == '1':
                    observable = observable + beta[n-1-i, n-1-j]
                elif key[j] == '0' and key[i] == '1':
                    observable = observable - beta[n-1-i, n-1-j]
                elif key[j] == '1' and key[i] == '0':
                    observable = observable - beta[n-1-i, n-1-j]
        temp += data[key]*observable/tot
    return temp
