# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
================================================
Quantum Information (:mod:`qiskit.quantum_info`)
================================================

.. currentmodule:: qiskit.quantum_info

Overview
========

**Quantum information science** combines quantum mechanics and information theory to represent, 
process, and transmit information using quantum states. 

In Qiskit, quantum information representations and manipulations are performed 
using the :class:`quantum_info` class. 

.. _quantum_info_operators:

Operators
=========

The matrix :class:`0perator` class represents an operator :math:`M` that will 
evolve a :class:`Statevector` :math:`|\\psi\\rangle` via matrix-vector multiplication

.. math::

   |\\psi\\rangle\\mapsto M|\\psi\\rangle,


and will evolve a :class:`DensityMatrix` :math:`\\rho` by left and right multiplication,

.. math::

   \\rho \\mapsto M \\rho M^\\dagger.

An operator can be defined by providing a list of its entries. For example,

.. code-block:

   import qiskit.quantum_info as qi

   XX = qi.Operator([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])
   print(XX)

.. parsed-literal::

   Operator([[0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
            [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
            [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
            [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]],
            input_dims=(2, 2), output_dims=(2, 2))

or by passing a :class:`QuantumCircuit` class, 

.. code-block:

   from qiskit import QuantumCircuit
   import qiskit.quantum_info as qi

   qc = QuantumCircuit(2)
   qc.x(0)
   qc.x(1)

   XX = qi.Operator(qc)
   print(XX)

.. parsed-literal::

   Operator([[0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
            [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
            [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
            [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]],
            input_dims=(2, 2), output_dims=(2, 2))


Similarly, :class:`Pauli` representations can be used, 

.. code-block:

   from qiskit import QuantumCircuit
   import qiskit.quantum_info as qi

   XX = qi.Operator(qi.Pauli('XX'))
   print(XX)

.. parsed-literal::

   Operator([[0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
            [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
            [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
            [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]],
            input_dims=(2, 2), output_dims=(2, 2))

.. autosummary::
   :toctree: ../stubs/

   Operator
   Pauli
   Clifford
   ScalarOp
   SparsePauliOp
   CNOTDihedral
   PauliList
   PauliTable
   StabilizerTable
   pauli_basis

.. _quantum_info_states:

States
======

Inside Qiskit, quantum information can be represented in different ways, including 
statevectors and density matrices. For example, the state:

.. math::

   |\\psi\\rangle = \\left(|0\\rangle+|1\\rangle\\right)/\\sqrt{2},

can be created in Qiskit using a (normalized) list of entries, 

.. code-block::

   import numpy as np
   import qiskit.quantum_info as qi

   statevector = qi.Statevector([1,0,0,1]/np.sqrt(2))
   print(statevector)

.. parsed-literal::

   Statevector([0.70710678+0.j, 0.        +0.j, 
                0.        +0.j, 0.70710678+0.j],
                dims=(2, 2))

Alternatively, this statevector can be obtained using a quantum circuit:

.. code-block::

   from qiskit import QuantumCircuit
   import qiskit.quantum_info as qi

   qc = QuantumCircuit(2)
   qc.h(0)
   qc.cx(0,1)

   statevector = qi.Statevector(qc)
   print(statevector)

.. parsed-literal::

   Statevector([0.70710678+0.j, 0.        +0.j, 
                0.        +0.j, 0.70710678+0.j],
                dims=(2, 2))
               

This statevector can also be directly represented using density matrices. To 
write the density matrix of the state above, for example, we can use the :class:`DensityMatrix`:

.. code-block::

   from qiskit import QuantumCircuit
   import qiskit.quantum_info as qi

   qc = QuantumCircuit(2)
   qc.h(0)
   qc.cx(0,1)

   densitymatrix = qi.DensityMatrix (qc)
   print(densitymatrix)

.. parsed-literal::

   Statevector([0.70710678+0.j, 0.        +0.j, 
                0.        +0.j, 0.70710678+0.j],
                dims=(2, 2))


.. autosummary::
   :toctree: ../stubs/

   Statevector
   DensityMatrix
   StabilizerState

Channels
========

Quantum channels are communication channels which can transmit quantum and
classical information. In Qiskit, different channel representations are implemented
by different classes. Similarly to operators, these channels also contain matrix methods 
such as :meth:`adjoint`, :meth:`compose`, and :meth:`conjugate`, among others. 

For example, one can obtain the Kraus representation of the quantum
channel modeled by the circuit below,

.. plot::
   :include-source:

   from qiskit import QuantumCircuit
   import qiskit.quantum_info as qi

   qc = QuantumCircuit(2)
   qc.cx(1,0)
   # Draw the circuit
   qc.draw('mpl')

.. code-block::

   kraus_rep = qi.Kraus(qc)
   print(kraus_rep)

.. parsed-literal::

   Kraus([[[-1.-0.j,  0.+0.j,  0.+0.j,  0.+0.j],
         [ 0.+0.j, -1.-0.j,  0.+0.j,  0.+0.j],
         [ 0.+0.j,  0.+0.j,  0.+0.j, -1.-0.j],
         [ 0.+0.j,  0.+0.j, -1.-0.j,  0.+0.j]]],
         input_dims=(2, 2), output_dims=(2, 2))

.. autosummary::
   :toctree: ../stubs/

   Choi
   SuperOp
   Kraus
   Stinespring
   Chi
   PTM

Measures
========

Different quantum information measures are included in Qiskit, including
measures of state and gate fidelity, useful for noisy calculations, 
purity, and mutual information of bipartite states, to name a few. A full list
of quantum information measures implemented in Qiskit is included below.

For example,

.. code-block::

   from qiskit import QuantumCircuit
   import qiskit.quantum_info as qi

   densitymatrix = (
      1/2*qi.DensityMatrix.from_label('+') +
      1/2*qi.DensityMatrix.from_label('0'))

   qc = QuantumCircuit(1)
   qc.x(0)

   evolved_dm = densitymatrix.evolve(qc)
   purity_evolved_dm = qi.purity(evolved_dm)
   print(purity_evolved_dm)

.. parsed-literal::
 
   (0.75+0j)

   
.. autosummary::
   :toctree: ../stubs/

   average_gate_fidelity
   process_fidelity
   gate_error
   diamond_norm
   state_fidelity
   purity
   concurrence
   entropy
   entanglement_of_formation
   mutual_information

Utility Functions
=================

Several useful tools is quantum information science, such as the partial
trace of a density operator or the Shannon entropy can be calculated using
the :class:`quantum_info` class. FOr example,

.. code-block::

   import qiskit.quantum_info as qi

   densitymatrix = (
      1/2*qi.DensityMatrix.from_label('+1') +
      1/2*qi.DensityMatrix.from_label('01'))

   reduced_dm = qi.partial_trace(densitymatrix, [0])
   print(reduced_dm)

.. parsed-literal::

   DensityMatrix([[0.75+0.j, 0.25+0.j],
                  [0.25+0.j, 0.25+0.j]],
                  dims=(2,))

.. autosummary::
   :toctree: ../stubs/

   partial_trace
   shannon_entropy
   commutator
   anti_commutator
   double_commutator

Random
======

It is also possible to use random representations of statevectors, 
density matrices, and operators in Qiskit.

Examples:

.. block-code::

   import qiskit.quantum_info as qi

   random_sv = qi.random_statevector(dims=4)
   print(random_sv)

.. parsed-literal::

   Statevector([-0.00896125-0.5442568j , -0.14707071-0.31546866j,
               0.28714761+0.14425677j,  0.66857182+0.17972737j],
               dims=(2, 2))

.. block-code::

   import qiskit.quantum_info as qi

   random_sv = qi.random_pauli(num_qubits=4)
   print(random_sv)

.. parsed-literal::

   YYXZ


.. autosummary::
   :toctree: ../stubs/

   random_statevector
   random_density_matrix
   random_unitary
   random_hermitian
   random_pauli
   random_clifford
   random_quantum_channel
   random_cnotdihedral
   random_pauli_table
   random_pauli_list
   random_stabilizer_table

Analysis
=========

Qiskit also permits to compute the Hellinger distance and
fidelity, which compares counts instead of statevectors. For example,

.. code-block::
   import qiskit.quantum_info as qi

   counts0 ={'00':489, '01':250, '10':250, '11':11}
   counts1 ={'00':389, '01':150, '10':350, '11':111}


   h_dist = qi.hellinger_distance(counts0, counts1)
   print(h_dist)

.. parsed-literal::

   0.19864719491829705

.. autosummary::
   :toctree: ../stubs/

   hellinger_distance
   hellinger_fidelity
   Z2Symmetries

Synthesis
=========

In Qiskit it is also possible to create decompositions of 
gates using different bases. A two-qubit gate decomposition
using the Euler basis `ZYZ` can be obtained, for example:

.. code-block::
   from qiskit import QuantumCircuit
   from qiskit.converters import circuit_to_gate
   import qiskit.quantum_info as qi

   qc = QuantumCircuit(2)
   qc.h(0)
   qc.cx(0,1)

   # convert circuit to gate
   qc_gate = circuit_to_gate(qc)
   qc_gate_ZYZ = qi.TwoQubitBasisDecomposer(qc_gate, euler_basis="ZYZ")
   print(qc_gate_ZYZ.basis)

.. parsed-literal::

   TwoQubitWeylControlledEquiv(
      global phase: 7π/4
         ┌──────────┐┌───────┐┌────────────┐┌─────────┐┌─────────┐
      q_0: ┤ Rx(-π/4) ├┤ Ry(π) ├┤0           ├┤ Rx(π/4) ├┤ Ry(π/2) ├
         ├──────────┤├───────┤│  Rxx(-π/2) │├─────────┤└┬───────┬┘
      q_1: ┤ Rx(3π/4) ├┤ Ry(π) ├┤1           ├┤ Rx(π/4) ├─┤ Ry(π) ├─
         └──────────┘└───────┘└────────────┘└─────────┘ └───────┘ 
)


.. autosummary::
   :toctree: ../stubs/

   OneQubitEulerDecomposer
   TwoQubitBasisDecomposer
   two_qubit_cnot_decompose
   Quaternion
   decompose_clifford
   XXDecomposer
"""

from .analysis import hellinger_distance, hellinger_fidelity, Z2Symmetries
from .operators import (
    Clifford,
    Operator,
    Pauli,
    PauliList,
    PauliTable,
    ScalarOp,
    SparsePauliOp,
    StabilizerTable,
    anti_commutator,
    commutator,
    double_commutator,
    pauli_basis,
)
from .operators.channel import PTM, Chi, Choi, Kraus, Stinespring, SuperOp
from .operators.dihedral import CNOTDihedral
from .operators.measures import average_gate_fidelity, diamond_norm, gate_error, process_fidelity
from .random import (
    random_clifford,
    random_cnotdihedral,
    random_density_matrix,
    random_hermitian,
    random_pauli,
    random_pauli_list,
    random_pauli_table,
    random_quantum_channel,
    random_stabilizer_table,
    random_statevector,
    random_unitary,
)
from .states import (
    DensityMatrix,
    StabilizerState,
    Statevector,
    concurrence,
    entanglement_of_formation,
    entropy,
    mutual_information,
    partial_trace,
    purity,
    shannon_entropy,
    state_fidelity,
)
from .synthesis import (
    OneQubitEulerDecomposer,
    Quaternion,
    TwoQubitBasisDecomposer,
    XXDecomposer,
    decompose_clifford,
    two_qubit_cnot_decompose,
)
