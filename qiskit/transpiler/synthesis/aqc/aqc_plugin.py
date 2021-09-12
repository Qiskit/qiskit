# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
An AQC synthesis plugin to Qiskit's transpiler.
"""
import abc

import numpy as np

from qiskit.algorithms.optimizers import L_BFGS_B
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.synthesis.aqc.aqc import AQC
from qiskit.transpiler.synthesis.aqc.cnot_structures import make_cnot_network

# todo: this is a copy from the PR: https://github.com/Qiskit/qiskit-terra/pull/6124
from qiskit.transpiler.synthesis.aqc.cnot_unit_circuit import CNOTUnitCircuit
from qiskit.transpiler.synthesis.aqc.cnot_unit_objective import DefaultCNOTUnitObjective


class UnitarySynthesisPlugin(abc.ABC):
    """
    Abstract plugin Synthesis plugin class.
    This abstract class defines the interface for unitary synthesis plugins.
    """

    @property
    @abc.abstractmethod
    def supports_basis_gates(self):
        """Return whether the plugin supports taking basis_gates"""
        pass

    @property
    @abc.abstractmethod
    def supports_coupling_map(self):
        """Return whether the plugin supports taking coupling_map"""
        pass

    @property
    @abc.abstractmethod
    def supports_approximation_degree(self):
        """Return whether the plugin supports taking approximation_degree"""
        pass

    @abc.abstractmethod
    def run(self, unitary, **options):
        """Run synthesis for the given unitary matrix
        Args:
            unitary (numpy.ndarray): The unitary matrix to synthesize to a
                :class:`~qiskit.dagcircuit.DAGCircuit` object
            options: The optional kwargs that are passed based on the output
                of :meth:`supports_basis_gates`, :meth:`supports_coupling_map`,
                and :meth:`supports_approximation_degree`. If
                :meth:`supports_coupling_map` returns ``True`` a kwarg
                ``coupling_map`` will be passed either containing ``None`` (if
                there is no coupling map) or a
                :class:`~qiskit.transpiler.CouplingMap` object. If
                :meth:`supports_basis_gates` returns ``True`` then a kwarg
                ``basis_gates`` will the list of basis gate names will be
                passed. Finally if :meth:`supports_approximation_degree`
                returns ``True`` a kwarg ``approximation_degree`` containing
                a float for the approximation value will be passed.
        Returns:
            DAGCircuit: The dag circuit representation of the unitary
        """
        pass


class AQCSynthesisPlugin(UnitarySynthesisPlugin):
    """An AQC-based Qiskit unitary synthesis plugin."""

    @property
    def supports_basis_gates(self):
        return False

    @property
    def supports_coupling_map(self):
        return False

    @property
    def supports_approximation_degree(self):
        return True

    def run(self, unitary, **options):
        num_qubits = int(round(np.log2(unitary.shape[0])))

        layout = options.get("layout") or "spin"
        connectivity = options.get("connectivity") or "full"
        depth = int(options.get("depth") or 0)
        cnots = make_cnot_network(
            num_qubits=num_qubits,
            network_layout=layout,
            connectivity_type=connectivity,
            depth=depth,
        )

        seed = options.get("seed")
        max_iter = options.get("max_iter") or 1000
        optimizer = L_BFGS_B(maxiter=max_iter)
        aqc = AQC(optimizer, seed)

        tol = options.get("approximation_degree") or 0.01
        name = options.get("approx_name") or "aqc"
        approximate_circuit = CNOTUnitCircuit(num_qubits, cnots, tol, name)
        approximating_objective = DefaultCNOTUnitObjective(num_qubits, cnots)

        aqc.compile_unitary(
            target_matrix=unitary,
            approximate_circuit=approximate_circuit,
            approximating_objective=approximating_objective,
        )

        dag_circuit = circuit_to_dag(approximate_circuit)
        return dag_circuit
