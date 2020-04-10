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

"""
=====================================
Transpiler (:mod:`qiskit.transpiler`)
=====================================

.. currentmodule:: qiskit.transpiler

Overview
========
Transpilation is the process of rewriting a given input circuit to match
the topoplogy of a specific quantum device, and/or to optimize the circuit
for execution on present day noisy quantum systems.

Most circuits must undergo a series of transformations that make them compatible with
a given target device, and optimize them to reduce the effects of noise on the
resulting outcomes.  Rewriting quantum circuits to match hardware constraints and
optimizing for performance can be far from trivial.  The flow of logic in the rewriting
tool chain need not be linear, and can often have iterative sub-loops, conditional
branches, and other complex behaviors.  That being said, the basic building blocks
follow the structure given below:

.. image:: /source_images/transpiling_core_steps.png

.. raw:: html

   <br><br>


Supplementary Information
=========================

.. container:: toggle

   .. container:: header

      **Basis Gates**

   When writing a quantum circuit you are free to use any quantum gate (unitary operator) that
   you like, along with a collection of non-gate operations such as qubit measurements and
   reset operations.  However, when running a circuit on a real quantum device one no longer
   has this flexibility.  Due to limitations in, for example, the physical interactions
   between qubits, difficulty in implementing multi-qubit gates, control electronics etc,
   a quantum computing device can only natively support a handful of quantum gates and non-gate
   operations. In the present case of IBM Q devices, the native gate set can be found by querying
   the devices themselves, and looking for the corresponding attribute in their configuration:

   .. jupyter-execute::
      :hide-code:
      :hide-output:

      from qiskit.test.mock import FakeVigo
      backend = FakeVigo()


   .. jupyter-execute::

      backend.configuration().basis_gates

   We see that IBM Quantum devices support five native gates: four single-qubit gates
   [`u1`, `u2`, `u3`, and `id`] and one two-qubit entangling gate `cx`.  In addition, the
   devices support qubit measurements (otherwise we can not read out an answer!).

   The `u*` gates represent arbitrary single-qubit rotations of one, two, and three angles.
   The `u1` gates are single-parameter rotations that represent generalized phase gates of
   the form:

   .. math::

      U_{1}(\\lambda) = \\begin{bmatrix}
      1 & 0 \\\\
      0 & e^{i\\lambda}
      \\end{bmatrix}

   This set includes common gates such as :math:`Z`, :math:`T`, :math:`T^{\\dagger}`, :math:`S`,
   and :math:`S^{\\dagger}`.  It turns out that these gates do not actually need to be performed
   on hardware, but instead, can be implemented in software as "virtual gates".  These virtual
   gates are called "frame changes" and take zero time, and have no associated error; they are
   free gates on hardware.

   Two-angle rotations, :math:`U_{2}(\\phi, \\lambda)`, are actually two frame changes with a single
   :math:`X_{\\pi/2}` gate in between them, and can be used to synthesize gates like the Hadamard
   (:math:`U_{2}(0,\\pi)`) gate.  As the only actual gate performed is the :math:`X_{\\pi/2}`
   gate, the error and gate time associated with any :math:`U_{2}` gate is the same as an
   :math:`X_{\\pi/2}` gate.  Similarly, :math:`U_{3}(\\theta,\\phi,\\lambda)` gates are formed
   from three frame changes with two :math:`X_{\\pi/2}` gates in between them.  The errors and
   gate times are twice those of a single :math:`X_{\\pi/2}`. The identity gate, `id`, is
   straightforward (although the qubit may not actually remain idle during this time), and is
   a placeholder gate with a fixed time-interval.

   The only entangling gate supported by the IBM Quantum devices is the CNOT gate (`cx`)
   which, in the computational basis, can be written as:

   .. math::

      \\mathrm{cx}(0, 1) = \\begin{bmatrix}
      1 & 0 & 0 & 0 \\\\
      0 & 0 & 0 & 1 \\\\
      0 & 0 & 1 & 0 \\\\
      0 & 1 & 0 & 0
      \\end{bmatrix}

   where the matrix form above follows from the specific bit-ordering convention used in Qiskit.

   Every quantum circuit run on an IBM Q device must be expressed using only these basis gates.
   For example, suppose one wants to run a simple phase estimation circuit:

   .. jupyter-execute::

      import numpy as np
      from qiskit import QuantumCircuit
      qc = QuantumCircuit(2, 1)

      qc.h(0)
      qc.x(1)
      qc.cu1(np.pi/4, 0, 1)
      qc.h(0)
      qc.measure([0], [0])
      qc.draw(output='mpl')

   We have :math:`H`, :math:`X`, and controlled-:math:`U_{1}` gates, all of which are
   not in our devices basis gate set, and must be expanded.  This expansion is taken
   care of for us in the :func:`qiskit.execute.execute` function. However, we can
   decompose the circuit to show what it would look like in the native gate set of
   the IBM Quantum devices:

   .. jupyter-execute::

      qc_basis = qc.decompose()
      qc_basis.draw(output='mpl')


   A few things to highlight.  First, the circuit has gotten longer with respect to the
   initial one.  This can be verified by checking the depth of the circuits:

   .. jupyter-execute::

      print('Original depth:', qc.depth(), 'Decomposed Depth:', qc_basis.depth())

   Second, although we had a single controlled gate, the fact that it was not in the basis
   set means that, when expanded, it requires more than a single `cx` gate to implement.
   All said, unrolling to the basis set of gates leads to an increase in the depth of a
   quantum circuit and the number of gates.

   It is important to highlight two special cases:

   1. A SWAP gate is not a native gate on the IBM Q devices, and must be decomposed into
      three CNOT gates:

      .. jupyter-execute::

         swap_circ = QuantumCircuit(2)
         swap_circ.swap(0, 1)
         swap_circ.decompose().draw(output='mpl')

      As a product of three CNOT gates, SWAP gates are expensive operations to perform on a
      noisy quantum devices.  However, such operations are usually necessary for embedding a
      circuit into the limited entangling gate connectivities of actual devices.  Thus,
      minimizing the number of SWAP gates in a circuit is a primary goal in the
      transpilation process.


   2. A Toffoli, or controlled-controlled-not gate (`ccx`), is a three-qubit gate.  Given
      that our basis gate set includes only single- and two-qubit gates, it is obvious that
      this gate must be decomposed.  This decomposition is quite costly:

      .. jupyter-execute::

         ccx_circ = QuantumCircuit(3)
         ccx_circ.ccx(0, 1, 2)
         ccx_circ.decompose().draw(output='mpl')

      For every Toffoli gate in a quantum circuit, the IBM Quantum hardware may execute up to
      six CNOT gates, and a handful of single-qubit gates.  From this example, it should be
      clear that any algorithm that makes use of multiple Toffoli gates will end up as a
      circuit with large depth and will therefore be appreciably affected by noise and gate
      errors.

   .. raw:: html

      <br>

.. container:: toggle

   .. container:: header

      **Initial Layout**

   Quantum circuits are abstract entities whose qubits are "virtual" representations of actual
   qubits used in computations.  We need to be able to map these virtual qubits in a one-to-one
   manner to the "physical" qubits in an actual quantum device.

   .. image:: /source_images/mapping.png

   .. raw:: html

      <br><br>

   By default, qiskit will do this mapping for you.  The choice of mapping depends on the
   properties of the circuit, the particular device you are targeting, and the optimization
   level that is chosen.  The basic mapping strategies are the following:

   - **Trivial layout**: Map virtual qubits to the same numbered physical qubit on the device,
   i.e. `[0,1,2,3,4]` -> `[0,1,2,3,4]` (default in `optimization_level=0`).

   - **Dense layout**: Find the sub-graph of the device with same number of qubits as the circuit
   with the greatest connectivity (default in `optimization_level=1`).

   - **Noise adaptive layout**: Uses the noise properties of the device, in concert with the
   circuit properties, to generate the layout with the best noise properties (default in
   `optimization_level=2` and `optimization_level=3`).

   The choice of initial layout is extremely important when:

   1. Computing the number of SWAP operations needed to map the input circuit onto the device
      topology.

   2. Taking into account the noise properties of the device.

   The choice of `initial_layout` can mean the difference between getting a result,
   and getting nothing but noise.

   Lets see what layouts are automatically picked at various optimization levels.  The modified
   circuits returned by :func:`qiskit.compiler.transpile` have this initial layout information
   in them, and we can view this layout selection graphically using
   :func:`qiskit.visualization.plot_circuit_layout`.

   .. raw:: html

      <br>

Transpiler API
==============

Pass Management
---------------

.. autosummary::
   :toctree: ../stubs/

   PassManager
   PassManagerConfig
   PropertySet
   FlowController

Layout and Topology
-------------------

.. autosummary::
   :toctree: ../stubs/

   Layout
   CouplingMap

Fenced Objects
--------------

.. autosummary::
   :toctree: ../stubs/

   FencedDAGCircuit
   FencedPropertySet

Exceptions
----------

.. autosummary::
   :toctree: ../stubs/

   TranspilerError
   TranspilerAccessError
"""

from .runningpassmanager import FlowController
from .passmanager import PassManager
from .passmanager_config import PassManagerConfig
from .propertyset import PropertySet
from .exceptions import TranspilerError, TranspilerAccessError
from .fencedobjs import FencedDAGCircuit, FencedPropertySet
from .basepasses import AnalysisPass, TransformationPass
from .coupling import CouplingMap
from .layout import Layout
