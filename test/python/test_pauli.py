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
"""Quick program to test the Pauli class."""
import numpy as np
from scipy import linalg as la
import sys
sys.path.append("../..")
from tools.pauli import Pauli, random_pauli, inverse_pauli, pauli_group, sgn_prod

v = np.zeros(3)
w = np.zeros(3)
v[0] = 1
w[1] = 1
v[2] = 1
w[2] = 1

p = Pauli(v, w)
print(p)
print("In label form:")
print(p.to_label())
print("In matrix form:")
print(p.to_matrix())


q = random_pauli(2)
print(q)

r = inverse_pauli(p)
print("In label form:")
print(r.to_label())

print("Group in tensor order:")
grp = pauli_group(3, case=1)
for j in grp:
    print(j.to_label())

print("Group in weight order:")
grp = pauli_group(3)
for j in grp:
    print(j.to_label())

print("sign product:")
p1 = Pauli(np.array([0]), np.array([1]))
p2 = Pauli(np.array([1]), np.array([1]))
p3, sgn = sgn_prod(p1, p2)
print(p1.to_label())
print(p2.to_label())
print(p3.to_label())
print(sgn)

print("sign product reverse:")
p3, sgn = sgn_prod(p2, p1)
print(p2.to_label())
print(p1.to_label())
print(p3.to_label())
print(sgn)
