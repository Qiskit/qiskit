# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Helper module for simplified Qiskit usage.

The functions in this module provide convenience converters
"""

from .circuit_to_dag import circuit_to_dag
from .dag_to_circuit import dag_to_circuit
from .ast_to_dag import ast_to_dag
from .circuit_to_instruction import circuit_to_instruction
