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

"""Quick program to test json backend

python test_jsonoutput.py ../qasm/example.qasm

"""
import sys
import numpy as np
from qiskit import QuantumProgram
import qiskit.qasm as qasm
import qiskit.unroll as unroll


seed = 88
filename = sys.argv[1]
qp = QuantumProgram()
qp.load_qasm("example", qasm_file=filename)

basis_gates = []  # unroll to base gates
unroller = unroll.Unroller(qasm.Qasm(data=qp.get_qasm("example")).parse(),
                           unroll.JsonBackend(basis_gates))
unroller.execute()
print(unroller.backend.circuit)
