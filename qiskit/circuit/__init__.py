# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
========================================
Quantum Circuits (:mod:`qiskit.circuit`)
========================================

.. currentmodule:: qiskit.circuit

Overview
========

The fundamental element of quantum computing is the **quantum circuit**.
A quantum circuit is a computational routine consisting of coherent quantum
operations on quantum data, such as qubits. It is an ordered sequence of quantum
gates, measurements and resets, which may be conditioned on real-time classical
computation. A set of quantum gates is said to be universal if any unitary
transformation of the quantum data can be efficiently approximated arbitrarily well
as as sequence of gates in the set. Any quantum program can be represented by a
sequence of quantum circuits and classical near-time computation.

In Qiskit, this core element is represented by the :class:`QuantumCircuit` class.
Below is an example of a quantum circuit that makes a three-qubit GHZ state
defined as:

.. math::

   |\\psi\\rangle = \\left(|000\\rangle+|111\\rangle\\right)/\\sqrt{2}


.. jupyter-execute::

   from qiskit import QuantumCircuit
   # Create a circuit with a register of three qubits
   circ = QuantumCircuit(3)
   # H gate on qubit 0, putting this qubit in a superposition of |0> + |1>.
   circ.h(0)
   # A CX (CNOT) gate on control qubit 0 and target qubit 1 generating a Bell state.
   circ.cx(0, 1)
   # CX (CNOT) gate on control qubit 0 and target qubit 2 resulting in a GHZ state.
   circ.cx(0, 2)
   # Draw the circuit
   circ.draw()


Supplementary Information
=========================

.. dropdown:: Quantum Circuit Properties
   :animate: fade-in-slide-down

   When constructing quantum circuits, there are several properties that help quantify
   the "size" of the circuits, and their ability to be run on a noisy quantum device.
   Some of these, like number of qubits, are straightforward to understand, while others
   like depth and number of tensor components require a bit more explanation.  Here we will
   explain all of these properties, and, in preparation for understanding how circuits change
   when run on actual devices, highlight the conditions under which they change.

   Consider the following circuit:

   .. jupyter-execute::

      from qiskit import QuantumCircuit
      qc = QuantumCircuit(12)
      for idx in range(5):
         qc.h(idx)
         qc.cx(idx, idx+5)

      qc.cx(1, 7)
      qc.x(8)
      qc.cx(1, 9)
      qc.x(7)
      qc.cx(1, 11)
      qc.swap(6, 11)
      qc.swap(6, 9)
      qc.swap(6, 10)
      qc.x(6)
      qc.draw()

   From the plot, it is easy to see that this circuit has 12 qubits, and a collection of
   Hadamard, CNOT, X, and SWAP gates.  But how to quantify this programmatically? Because we
   can do single-qubit gates on all the qubits simultaneously, the number of qubits in this
   circuit is equal to the **width** of the circuit:

   .. jupyter-execute::

      qc.width()


   We can also just get the number of qubits directly:

   .. jupyter-execute::

      qc.num_qubits


   .. important::

      For a quantum circuit composed from just qubits, the circuit width is equal
      to the number of qubits. This is the definition used in quantum computing. However,
      for more complicated circuits with classical registers, and classically controlled gates,
      this equivalence breaks down. As such, from now on we will not refer to the number of
      qubits in a quantum circuit as the width.


   It is also straightforward to get the number and type of the gates in a circuit using
   :meth:`QuantumCircuit.count_ops`:

   .. jupyter-execute::

      qc.count_ops()


   We can also get just the raw count of operations by computing the circuits
   :meth:`QuantumCircuit.size`:

   .. jupyter-execute::

      qc.size()


   A particularly important circuit property is known as the circuit **depth**.  The depth
   of a quantum circuit is a measure of how many "layers" of quantum gates, executed in
   parallel, it takes to complete the computation defined by the circuit.  Because quantum
   gates take time to implement, the depth of a circuit roughly corresponds to the amount of
   time it takes the quantum computer to execute the circuit.  Thus, the depth of a circuit
   is one important quantity used to measure if a quantum circuit can be run on a device.

   The depth of a quantum circuit has a mathematical definition as the longest path in a
   directed acyclic graph (DAG).  However, such a definition is a bit hard to grasp, even for
   experts.  Fortunately, the depth of a circuit can be easily understood by anyone familiar
   with playing `Tetris <https://en.wikipedia.org/wiki/Tetris>`_.  Lets see how to compute this
   graphically:

   .. image:: /source_images/depth.gif


   .. raw:: html

      <br><br>


   We can verify our graphical result using :meth:`QuantumCircuit.depth`:

   .. jupyter-execute::

      qc.depth()


   .. raw:: html

      <br>

Quantum Circuit API
===================

Quantum Circuit Construction
----------------------------

.. autosummary::
   :toctree: ../stubs/

   QuantumCircuit
   QuantumRegister
   Qubit
   ClassicalRegister
   Clbit
   AncillaRegister
   AncillaQubit

Gates and Instructions
----------------------

.. autosummary::
   :toctree: ../stubs/

   Gate
   ControlledGate
   Delay
   Measure
   Reset
   Instruction
   InstructionSet
   EquivalenceLibrary

Parametric Quantum Circuits
---------------------------

.. autosummary::
   :toctree: ../stubs/

    Parameter
    ParameterVector
    ParameterExpression

Random Circuits
---------------

.. autosummary::
   :toctree: ../stubs/

   random.random_circuit
"""
from .quantumcircuit import QuantumCircuit
from .classicalregister import ClassicalRegister, Clbit
from .quantumregister import QuantumRegister, Qubit, AncillaRegister, AncillaQubit
from .gate import Gate

# pylint: disable=cyclic-import
from .controlledgate import ControlledGate
from .instruction import Instruction
from .instructionset import InstructionSet
from .barrier import Barrier
from .delay import Delay
from .measure import Measure
from .reset import Reset
from .parameter import Parameter
from .parametervector import ParameterVector
from .parameterexpression import ParameterExpression
from .equivalence import EquivalenceLibrary
from .classicalfunction.types import Int1, Int2
from .classicalfunction import classical_function, BooleanExpression
