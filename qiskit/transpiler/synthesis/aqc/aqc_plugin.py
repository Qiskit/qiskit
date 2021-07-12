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

from qiskit.converters import circuit_to_dag
from qiskit.transpiler.synthesis.aqc.aqc import AQC
from qiskit.transpiler.synthesis.aqc.cnot_structures import make_cnot_network


# todo: this is a copy from the PR: https://github.com/Qiskit/qiskit-terra/pull/6124
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
        approximation_degree = options.get("approximation_degree") or 0.01
        thetas = options.get("thetas")
        num_qubits = int(round(np.log2(unitary.shape[0])))
        aqc = AQC(
            method="nesterov",
            maxiter=1000,
            eta=0.01,
            tol=approximation_degree,
            eps=0.0,
        )

        cnots = make_cnot_network(
            num_qubits=num_qubits, network_layout="spin", connectivity_type="full"
        )

        parametric_circuit = aqc.compile_unitary(target_matrix=unitary, cnots=cnots, thetas=thetas)

        dag_circuit = circuit_to_dag(parametric_circuit.to_circuit())
        return dag_circuit
