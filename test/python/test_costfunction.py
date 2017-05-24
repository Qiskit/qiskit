# -*- coding: utf-8 -*-

# Copyright 2017 IBM RESEARCH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""Quick program to test the cost function for different basis functions.


Author: Jay Gambetta

    python test_simulators.py ../qasm/example.qasm qasm_simulator

"""
import numpy as np
from scipy import linalg as la
import sys
sys.path.append("../..")
from tools.optimizationtools import cost_function, make_Hamiltonian

n = 3
# cost function
alpha = np.zeros(n)
alpha[2] = 1
# only input the upper triangle  b[i,j] for i < j
beta = np.zeros((n, n))
beta[0, 1] = 1
beta[0, 2] = 1
beta[1, 2] = 1
a = make_Hamiltonian(n, alpha, beta)
print(a)
w, v = la.eigh(a, eigvals=(0, 0))
print(w)
print(v)
data = {'000': 10}
print(cost_function(data, n, alpha, beta))
data = {'001': 10}
print(cost_function(data, n, alpha, beta))
data = {'010': 10}
print(cost_function(data, n, alpha, beta))
data = {'011': 10}
print(cost_function(data, n, alpha, beta))
data = {'100': 10}
print(cost_function(data, n, alpha, beta))
data = {'101': 10}
print(cost_function(data, n, alpha, beta))
data = {'110': 10}
print(cost_function(data, n, alpha, beta))
data = {'111': 10}
print(cost_function(data, n, alpha, beta))
"""
[[ 4.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j -2.+0.j  0.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j -2.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j -2.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  2.+0.j]]
[-2.]
[[ 0.+0.j]
 [ 0.+0.j]
 [ 0.+0.j]
 [ 0.+0.j]
 [ 0.+0.j]
 [ 0.+0.j]
 [ 1.+0.j]
 [ 0.+0.j]]
4.0
0.0
0.0
0.0
-2.0
-2.0
-2.0
2.0
"""
