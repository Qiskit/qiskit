#%%

from numpy import array
import numpy as np


def match_case(m0, m1, n0, n1):
    if m0 == n0 and m1 == n1:
        return 2
    elif (m0 == n0 and m1 != n1) or (m0 != n0 and m1 == n1):
        return 1
    elif (m0 == n1 or m1 == n0):
        return -1
    elif m0 != n0 and m1 != n1:
        return 0
    else:
        raise AssertionError

def which_field(item):
    n0 = item[0]
    n1 = item[1]
    if n0[0] == n1[0]:
        if n0[1] != n1[1]:
            return 0
        raise AssertionError
    elif n0[1] == n1[1]:
        return 1
    print(item)
    raise AssertionError

def mat_solver(A, B):
    return np.linalg.solve(A, B)

def valid_transition(state1, state2):
    if state1[0] == state2[0]:
        if state1[1] - state2[1] == -1:
            return True
    elif state1[1] == state2[1]:
        if state1[0] - state2[0] == -1:
            return True
    return False

def main(states):
    linear_matrix = []
    row_labels = []
    for n0 in states:
        for n1 in states:
            if valid_transition(n0, n1):
                row = []
                row_labels.append((n0, n1))
                col_labels = []
                for m0 in states:
                    for m1 in states:
                        if valid_transition(m0, m1):
                            row.append(match_case(m0, m1, n0, n1))
                            col_labels.append((m0,m1))
                linear_matrix.append(row)
    B_0 = [1 - which_field(row) for row in row_labels] 
    B_1 = [which_field(row) for row in row_labels]
    field1R = mat_solver(linear_matrix, B_1)
    field0R = mat_solver(linear_matrix, B_0)
    
    assert(row_labels==col_labels)

    return linear_matrix, field0R, field1R, row_labels
    # return linear_matrix, B_0, B_1, row_labels

#%%
states = [(a,b) for a in range(2) for b in range(2)]
# %%
matrix, res0, res1, rows = main(states)
# %%
states
rows
# %%

# %%
