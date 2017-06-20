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

"""
Unitary simulator.

Authors: Jesus Perez <jesusper@us.ibm.com>
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from qiskit import UnitarySimulator


QASM_PATH = os.path.join(os.path.dirname(__file__), '../../qasm/simple8qbit.qasm')
CIRCUIT = open(QASM_PATH, 'r').read()

print('Result')
print(UnitarySimulator(CIRCUIT).run())
