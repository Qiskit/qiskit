# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

r"""
=======================================================================================
Approximately decompose 1q gates to a discrete basis using the Solovay-Kitaev algorithm
=======================================================================================

.. currentmodule:: qiskit.synthesis.discrete_basis

Approximately decompose 1q gates to a discrete basis using the Solovay-Kitaev algorithm.

The Solovay-Kitaev theorem [1] states that any single qubit gate can be approximated to
arbitrary precision by a set of fixed single-qubit gates, if the set generates a dense
subset in :math:`SU(2)`. This is an important result, since it means that any single-qubit
gate can be expressed in terms of a discrete, universal gate set that we know how to implement
fault-tolerantly. Therefore, the Solovay-Kitaev algorithm allows us to take any
non-fault tolerant circuit and rephrase it in a fault-tolerant manner.

This implementation of the Solovay-Kitaev algorithm is based on [2].

For example, the following circuit

.. parsed-literal::

         ┌─────────┐
    q_0: ┤ RX(0.8) ├
         └─────────┘

can be decomposed into

.. parsed-literal::

    global phase: 7π/8
         ┌───┐┌───┐┌───┐
    q_0: ┤ H ├┤ T ├┤ H ├
         └───┘└───┘└───┘

with an L2-error of approximately 0.01.


Examples:

    .. jupyter-execute::

        import numpy as np
        from qiskit.circuit import QuantumCircuit
        from qiskit.circuit.library import TGate, HGate, TdgGate
        from qiskit.converters import circuit_to_dag, dag_to_circuit
        from qiskit.transpiler.passes.synthesis import SolovayKitaev
        from qiskit.quantum_info import Operator

        circuit = QuantumCircuit(1)
        circuit.rx(0.8, 0)
        dag = circuit_to_dag(circuit)

        print('Original circuit:')
        print(circuit.draw())

        basis_gates = [TGate(), TdgGate(), HGate()]
        skd = SolovayKitaev(recursion_degree=2)

        discretized = dag_to_circuit(skd.run(dag))

        print('Discretized circuit:')
        print(discretized.draw())

        print('Error:', np.linalg.norm(Operator(circuit).data - Operator(discretized).data))


References:

    [1]: Kitaev, A Yu (1997). Quantum computations: algorithms and error correction.
         Russian Mathematical Surveys. 52 (6): 1191–1249.
         `Online <https://iopscience.iop.org/article/10.1070/RM1997v052n06ABEH002155>`_.

    [2]: Dawson, Christopher M.; Nielsen, Michael A. (2005) The Solovay-Kitaev Algorithm.
         `arXiv:quant-ph/0505030 <https://arxiv.org/abs/quant-ph/0505030>`_

"""

from .solovay_kitaev import SolovayKitaevDecomposition
