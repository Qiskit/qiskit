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
as a sequence of gates in the set. Any quantum program can be represented by a
sequence of quantum circuits and classical near-time computation.

In Qiskit, this core element is represented by the :class:`QuantumCircuit` class.
Below is an example of a quantum circuit that makes a three-qubit GHZ state
defined as:

.. math::

   |\\psi\\rangle = \\left(|000\\rangle+|111\\rangle\\right)/\\sqrt{2}


.. plot::
   :include-source:

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
   circ.draw('mpl')


Supplementary Information
=========================

.. dropdown:: Quantum Circuit with conditionals
   :animate: fade-in-slide-down

   When building a quantum circuit, there can be interest in applying a certain gate only
   if a classical register has a specific value. This can be done with the
   :meth:`InstructionSet.c_if` method.

   In the following example, we start with a single-qubit circuit formed by only a Hadamard gate
   (:class:`~.HGate`), in which we expect to get :math:`|0\\rangle` and :math:`|1\\rangle`
   with equal probability.

   .. plot::
      :include-source:

      from qiskit import BasicAer, transpile, QuantumRegister, ClassicalRegister, QuantumCircuit

      qr = QuantumRegister(1)
      cr = ClassicalRegister(1)
      qc = QuantumCircuit(qr, cr)
      qc.h(0)
      qc.measure(0, 0)
      qc.draw('mpl')

   .. code-block::

      backend = BasicAer.get_backend('qasm_simulator')
      tqc = transpile(qc, backend)
      counts = backend.run(tqc).result().get_counts()

      print(counts)

   .. parsed-literal::

      {'0': 524, '1': 500}

   Now, we add an :class:`~.XGate` only if the value of the :class:`~.ClassicalRegister` is 0.
   That way, if the state is :math:`|0\\rangle`, it will be changed to :math:`|1\\rangle` and
   if the state is :math:`|1\\rangle`, it will not be changed at all, so the final state will
   always be :math:`|1\\rangle`.

   .. plot::
      :include-source:

      from qiskit import BasicAer, transpile, QuantumRegister, ClassicalRegister, QuantumCircuit

      qr = QuantumRegister(1)
      cr = ClassicalRegister(1)
      qc = QuantumCircuit(qr, cr)
      qc.h(0)
      qc.measure(0, 0)

      qc.x(0).c_if(cr, 0)
      qc.measure(0, 0)

      qc.draw('mpl')

   .. code-block::

      backend = BasicAer.get_backend('qasm_simulator')
      tqc = transpile(qc, backend)
      counts = backend.run(tqc).result().get_counts()

      print(counts)

   .. parsed-literal::

      {'1': 1024}

.. dropdown:: Quantum Circuit Properties
   :animate: fade-in-slide-down

   When constructing quantum circuits, there are several properties that help quantify
   the "size" of the circuits, and their ability to be run on a noisy quantum device.
   Some of these, like number of qubits, are straightforward to understand, while others
   like depth and number of tensor components require a bit more explanation.  Here we will
   explain all of these properties, and, in preparation for understanding how circuits change
   when run on actual devices, highlight the conditions under which they change.

   Consider the following circuit:

   .. plot::
      :include-source:

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
      qc.draw('mpl')

   From the plot, it is easy to see that this circuit has 12 qubits, and a collection of
   Hadamard, CNOT, X, and SWAP gates.  But how to quantify this programmatically? Because we
   can do single-qubit gates on all the qubits simultaneously, the number of qubits in this
   circuit is equal to the **width** of the circuit:

   .. code-block::

      qc.width()

   .. parsed-literal::

      12

   We can also just get the number of qubits directly:

   .. code-block::

      qc.num_qubits

   .. parsed-literal::

      12

   .. important::

      For a quantum circuit composed from just qubits, the circuit width is equal
      to the number of qubits. This is the definition used in quantum computing. However,
      for more complicated circuits with classical registers, and classically controlled gates,
      this equivalence breaks down. As such, from now on we will not refer to the number of
      qubits in a quantum circuit as the width.


   It is also straightforward to get the number and type of the gates in a circuit using
   :meth:`QuantumCircuit.count_ops`:

   .. code-block::

      qc.count_ops()

   .. parsed-literal::

      OrderedDict([('cx', 8), ('h', 5), ('x', 3), ('swap', 3)])

   We can also get just the raw count of operations by computing the circuits
   :meth:`QuantumCircuit.size`:

   .. code-block::

      qc.size()

   .. parsed-literal::

      19

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

   .. code-block::

      qc.depth()

   .. parsed-literal::

      9

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
   CircuitInstruction
   Register
   Bit

Gates and Instructions
----------------------

.. autosummary::
   :toctree: ../stubs/

   Gate
   ControlledGate
   Delay
   Instruction
   InstructionSet
   Operation
   EquivalenceLibrary

Control Flow Operations
-----------------------

.. autosummary::
   :toctree: ../stubs/

   ControlFlowOp
   IfElseOp
   WhileLoopOp
   ForLoopOp
   SwitchCaseOp
   BreakLoopOp
   ContinueLoopOp


The :class:`.SwitchCaseOp` also understands a special value:

.. py:data:: CASE_DEFAULT

   A special object that represents the "default" case of a switch statement.  If you use this as a
   case target, it must be the last case, and will match anything that wasn't already matched.  For
   example::

       from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
       from qiskit.circuit import SwitchCaseOp, CASE_DEFAULT

       body0 = QuantumCircuit(2, 2)
       body0.x(0)
       body1 = QuantumCircuit(2, 2)
       body1.z(0)
       body2 = QuantumCircuit(2, 2)
       body2.cx(0, 1)

       qr, cr = QuantumRegister(2), ClassicalRegister(2)
       qc = QuantumCircuit(qr, cr)
       qc.switch(cr, [(0, body0), (1, body1), (CASE_DEFAULT, body2)], qr, cr)

   When using the builder interface of :meth:`.QuantumCircuit.switch`, this can also be accessed as
   the ``DEFAULT`` attribute of the bound case-builder object, such as::

       from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

       qr, cr = QuantumRegister(2), ClassicalRegister(2)
       qc = QuantumCircuit(qr, cr)
       with qc.switch(cr) as case:
           with case(0):
               qc.x(0)
           with case(1):
               qc.z(0)
           with case(case.DEFAULT):
               qc.cx(0, 1)


Parametric Quantum Circuits
---------------------------

.. autosummary::
   :toctree: ../stubs/

   Parameter
   ParameterVector
   ParameterExpression

Random Circuits
---------------

.. currentmodule:: qiskit.circuit.random
.. autofunction:: random_circuit
.. currentmodule:: qiskit.circuit

Circuit Analysis
----------------

.. autosummary::
   :toctree: ../stubs/

   CommutationChecker

Annotated Operations
--------------------

.. autosummary::
   :toctree: ../stubs/

   AnnotatedOperation
   ControlModifier
   PowerModifier
   InverseModifier

Exceptions
----------

Almost all circuit functions and methods will raise a :exc:`CircuitError` when encountering an error
that is particular to usage of Qiskit (as opposed to regular typing or indexing problems, which will
typically raise the corresponding standard Python error).

.. autoexception:: CircuitError
"""

from .exceptions import CircuitError
from .quantumcircuit import QuantumCircuit
from .classicalregister import ClassicalRegister, Clbit
from .quantumregister import QuantumRegister, Qubit, AncillaRegister, AncillaQubit
from .gate import Gate

# pylint: disable=cyclic-import
from .controlledgate import ControlledGate
from . import singleton
from .instruction import Instruction
from .instructionset import InstructionSet
from .operation import Operation
from .barrier import Barrier
from .delay import Delay
from .measure import Measure
from .reset import Reset
from .parameter import Parameter
from .parametervector import ParameterVector
from .parameterexpression import ParameterExpression
from .quantumcircuitdata import CircuitInstruction
from .equivalence import EquivalenceLibrary
from .bit import Bit
from .register import Register
from . import library
from .commutation_checker import CommutationChecker
from .annotated_operation import AnnotatedOperation, ControlModifier, PowerModifier, InverseModifier

from .controlflow import (
    ControlFlowOp,
    WhileLoopOp,
    ForLoopOp,
    IfElseOp,
    SwitchCaseOp,
    CASE_DEFAULT,
    BreakLoopOp,
    ContinueLoopOp,
)


_DEPRECATED_NAMES = {
    "Int1": "qiskit.circuit.classicalfunction.types",
    "Int2": "qiskit.circuit.classicalfunction.types",
    "classical_function": "qiskit.circuit.classicalfunction",
    "BooleanExpression": "qiskit.circuit.classicalfunction",
}


def __getattr__(name):
    if name in _DEPRECATED_NAMES:
        import importlib
        import warnings

        module_name = _DEPRECATED_NAMES[name]
        warnings.warn(
            f"Accessing '{name}' from '{__name__}' is deprecated since Qiskit Terra 0.22 "
            f"and will be removed in 0.23.  Import from '{module_name}' instead. "
            "This will require installing 'tweedledum' as an optional dependency from Terra 0.23.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(importlib.import_module(module_name), name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
