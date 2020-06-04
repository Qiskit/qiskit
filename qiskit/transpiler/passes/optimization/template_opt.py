# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
It extracts all maximal matches from the matches of the template matching algorithm.


The reference paper is https://arxiv.org/abs/1909.05270

"""
import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.converters.circuit_to_dagdependency import circuit_to_dagdependency
from qiskit.converters.dag_to_dagdependency import dag_to_dagdependency
from qiskit.converters.dag_to_circuit import dag_to_circuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.circuit.library.template_circuits.toffoli import *
from qiskit.quantum_info.operators.operator import Operator
from qiskit.transpiler.exceptions import TranspilerError

from qiskit.transpiler.passes.optimization.template_matching import *


class TemplateOptimization(TransformationPass):

    def __init__(self, template_list=None):
        super().__init__()

        if template_list is None:
            template_list = [template_2a_1(), template_2a_2(), template_2a_3()]
        self.template_list = template_list

    def run(self, circuit_dag):
        circuit_dag_dep = dag_to_dagdependency(circuit_dag)
        circuit = dag_to_circuit(circuit_dag)

        for template in self.template_list:
            if not isinstance(template, QuantumCircuit):
                raise TranspilerError('A template is a Quantumciruit().')

            identity = np.identity(2**template.num_qubits, dtype=complex)

            comparison = np.allclose(Operator(template).data, identity)

            if not comparison:
                raise TranspilerError('A template is a Quantumciruit() that performs the identity.')

            if template.num_qubits > len(circuit_dag_dep.qubits()):
                continue
            else:
                template_dag_dep = circuit_to_dagdependency(template)

                template_matching = TemplateMatching(circuit_dag_dep,template_dag_dep)
                template_matching.run_template_matching()

                matches = template_matching.match_list

            if matches:
                maximal_matches = MaximalMatches(matches)
                maximal_matches.run_maximal_matches()
                maximal_matches = maximal_matches.max_match_list

                substitution = TemplateSubstitution(maximal_matches, template_matching.circuit_dag_dep,
                                                    template_matching.template_dag_dep)
                substitution.run_dag_opt()

                circuit_dag_dep = substitution.dag_dep_optimized
                circuit_dag = substitution.dag_optimized
            else:
                continue

        if Operator(circuit) != Operator(dag_to_circuit(circuit_dag)):
            raise TranspilerError('A failure happened during the substitution')
        else:
            return circuit_dag
