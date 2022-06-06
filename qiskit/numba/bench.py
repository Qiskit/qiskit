# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Benchmark abs2 implementation in numba, vs. numpy implementation.
"""

import timeit
import numpy as np

def vector_setup(n):
    setup_code = f"""
import numpy as np
from qiskit.numba.fast_alternatives import abs2
n={n}; a = np.random.rand(n) + np.random.rand(n) * 1j
"""
    return setup_code


def matrix_setup(n):
    setup_code = f"""
import numpy as np
from qiskit.numba.fast_alternatives import abs2
n={n}; m = np.random.rand(n, n) + np.random.rand(n, n) * 1j
"""
    return setup_code


def timeit_avg(stmnt, setup=None, number=1000):
    result = timeit.timeit(stmnt, setup=setup, number=number)
    return result / number


def time_n_elem(stmnt, n_elem, ntrials, setup_func):
    setup_code = setup_func(n_elem)
    result = timeit_avg(stmnt, setup=setup_code, number=ntrials)
    return result


def run_trials(stmnt, setup_func):
    times = []
    for (n_elem, ntrials) in trial_pairs:
        time_n_elem(stmnt, n_elem, 1, setup_func)  # for jit compilation
        result = time_n_elem(stmnt, n_elem, ntrials, setup_func)
        times.append(result)
        print(f"{stmnt}, n_elem = {n_elem}, t = {result}")
    return times


print("\nVector")

trial_pairs = [
    (10, 10**5),
    (100, 10**5),
    (1000, 10**5),
    (10**4, 10**4),
    (10**5, 10**4),
    (10**6, 10**3),
]
stmnt = "abs2(a)"
abs2_times = run_trials(stmnt, vector_setup)

trial_pairs = [
    (10, 10**5),
    (100, 10**5),
    (1000, 10**5),
    (10**4, 10**4),
    (10**5, 10**4),
    (10**6, 10**3),
]
stmnt = "abs(a) ** 2"

abs_sq_times = run_trials(stmnt, vector_setup)

def print_ratios(times_new, times_old, trial_pairs):
    t_ratios = [t2 / t1 for (t1, t2) in zip(times_new, times_old)]
    for (n_elem, ratio) in zip([p[0] for p in trial_pairs], t_ratios):
        print(f"n_elem = 10**{int(np.log10(n_elem))}, t_ratio = {ratio}")

print_ratios(abs2_times, abs_sq_times, trial_pairs)


print("\nMatrix")

trial_pairs = [(10, 10**3), (100, 10**3), (1000, 10**2), (10**4, 10**2)]
stmnt = "abs2(m)"
abs2_times = run_trials(stmnt, matrix_setup)

trial_pairs = [(10, 10**3), (100, 10**3), (1000, 10**2), (10**4, 10**1)]
stmnt = "abs(m) ** 2"
abs_sq_times = run_trials(stmnt, matrix_setup)

print_ratios(abs2_times, abs_sq_times, trial_pairs)
