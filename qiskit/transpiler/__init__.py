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
the topology of a specific quantum device, and/or to optimize the circuit
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

   <br>

Qiskit has four pre-built transpilation pipelines available here:
:mod:`qiskit.transpiler.preset_passmanagers`.  Unless the reader is familiar with
quantum circuit optimization methods and their usage, it is best to use one of
these ready-made routines.


.. _transpiler_supplemental:

Supplementary Information
=========================

.. dropdown:: Basis gates
   :animate: fade-in-slide-down

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


   Every quantum circuit run on an IBM Q device must be expressed using only these basis gates.
   For example, suppose one wants to run a simple phase estimation circuit:

   .. jupyter-execute::

      import numpy as np
      from qiskit import QuantumCircuit
      qc = QuantumCircuit(2, 1)

      qc.h(0)
      qc.x(1)
      qc.cp(np.pi/4, 0, 1)
      qc.h(0)
      qc.measure([0], [0])
      qc.draw(output='mpl')

   We have :math:`H`, :math:`X`, and controlled-:math:`P` gates, all of which are
   not in our devices basis gate set, and must be expanded.  This expansion is taken
   care of for us in the :func:`qiskit.execute` function. However, we can
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

.. dropdown:: Initial layout
   :animate: fade-in-slide-down

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
     i.e. `[0,1,2,3,4]` -> `[0,1,2,3,4]` (default in `optimization_level=0` and
     `optimization_level=1`).

   - **Dense layout**: Find the sub-graph of the device with same number of qubits as the circuit
     with the greatest connectivity (default in `optimization_level=2` and `optimization_level=3`).


   The choice of initial layout is extremely important when:

   1. Computing the number of SWAP operations needed to map the input circuit onto the device
      topology.

   2. Taking into account the noise properties of the device.


   The choice of `initial_layout` can mean the difference between getting a result,
   and getting nothing but noise.

   Lets see what layouts are automatically picked at various optimization levels.  The modified
   circuits returned by :func:`qiskit.compiler.transpile` have this initial layout information
   in them, and we can view this layout selection graphically using
   :func:`qiskit.visualization.plot_circuit_layout`:

   .. jupyter-execute::

      from qiskit import QuantumCircuit, transpile
      from qiskit.visualization import plot_circuit_layout
      from qiskit.test.mock import FakeVigo
      backend = FakeVigo()

      ghz = QuantumCircuit(3, 3)
      ghz.h(0)
      ghz.cx(0,range(1,3))
      ghz.barrier()
      ghz.measure(range(3), range(3))
      ghz.draw(output='mpl')


   - **Layout Using Optimization Level 0**

      .. jupyter-execute::

         new_circ_lv0 = transpile(ghz, backend=backend, optimization_level=0)
         plot_circuit_layout(new_circ_lv0, backend)


   - **Layout Using Optimization Level 3**

      .. jupyter-execute::

         new_circ_lv3 = transpile(ghz, backend=backend, optimization_level=3)
         plot_circuit_layout(new_circ_lv3, backend)


   It is completely possible to specify your own initial layout.  To do so we can
   pass a list of integers to :func:`qiskit.compiler.transpile` via the `initial_layout`
   keyword argument, where the index labels the virtual qubit in the circuit and the
   corresponding value is the label for the physical qubit to map onto:

   .. jupyter-execute::

      # Virtual -> physical
      #    0    ->    3
      #    1    ->    4
      #    2    ->    2

      my_ghz = transpile(ghz, backend, initial_layout=[3, 4, 2])
      plot_circuit_layout(my_ghz, backend)

   .. raw:: html

      <br>


.. dropdown:: Mapping circuits to hardware topology
   :animate: fade-in-slide-down

   In order to implement a CNOT gate between qubits in a quantum circuit that are not directly
   connected on a quantum device one or more SWAP gates must be inserted into the circuit to
   move the qubit states around until they are adjacent on the device gate map.  Each SWAP
   gate is decomposed into three CNOT gates on the IBM Quantum devices, and represents an
   expensive and noisy operation to perform.  Thus, finding the minimum number of SWAP gates
   needed to map a circuit onto a given device, is an important step (if not the most important)
   in the whole execution process.

   However, as with many important things in life, finding the optimal SWAP mapping is hard.
   In fact it is in a class of problems called NP-Hard, and is thus prohibitively expensive
   to compute for all but the smallest quantum devices and input circuits.  To get around this,
   by default Qiskit uses a stochastic heuristic algorithm called
   :class:`Qiskit.transpiler.passes.StochasticSwap` to compute a good, but not necessarily minimal
   SWAP count.  The use of a stochastic method means the circuits generated by
   :func:`Qiskit.compiler.transpile` (or :func:`Qiskit.execute` that calls `transpile` internally)
   are not guaranteed to be the same over repeated runs.  Indeed, running the same circuit
   repeatedly will in general result in a distribution of circuit depths and gate counts at the
   output.

   In order to highlight this, we run a GHZ circuit 100 times, using a "bad" (disconnected)
   `initial_layout`:

   .. jupyter-execute::

      import matplotlib.pyplot as plt
      from qiskit import QuantumCircuit, transpile
      from qiskit.test.mock import FakeBoeblingen
      backend = FakeBoeblingen()

      ghz = QuantumCircuit(5)
      ghz.h(0)
      ghz.cx(0,range(1,5))
      ghz.draw(output='mpl')


   .. jupyter-execute::

      depths = []
      for _ in range(100):
         depths.append(transpile(ghz,
                                 backend,
                                 initial_layout=[7, 0, 4, 15, 19],
                                ).depth())

      plt.figure(figsize=(8, 6))
      plt.hist(depths, bins=list(range(14,36)), align='left', color='#AC557C')
      plt.xlabel('Depth', fontsize=14)
      plt.ylabel('Counts', fontsize=14);


   This distribution is quite wide, signaling the difficultly the SWAP mapper is having
   in computing the best mapping.  Most circuits will have a distribution of depths,
   perhaps not as wide as this one, due to the stochastic nature of the default SWAP
   mapper.  Of course, we want the best circuit we can get, especially in cases where
   the depth is critical to success or failure.  In cases like this, it is best to
   :func:`transpile` a circuit several times, e.g. 10, and take the one with the
   lowest depth.  The :func:`transpile` function will automatically run in parallel
   mode, making this procedure relatively speedy in most cases.

   .. raw:: html

      <br>


.. dropdown:: Gate optimization
   :animate: fade-in-slide-down

   Decomposing quantum circuits into the basis gate set of the IBM Quantum devices,
   and the addition of SWAP gates needed to match hardware topology, conspire to
   increase the depth and gate count of quantum circuits.  Fortunately many routines
   for optimizing circuits by combining or eliminating gates exist.  In some cases
   these methods are so effective the output circuits have lower depth than the inputs.
   In other cases, not much can be done, and the computation may be difficult to
   perform on noisy devices.  Different gate optimizations are turned on with
   different `optimization_level` values.  Below we show the benefits gained from
   setting the optimization level higher:

   .. important::

      The output from :func:`transpile` varies due to the stochastic swap mapper.
      So the numbers below will likely change each time you run the code.


   .. jupyter-execute::

      import matplotlib.pyplot as plt
      from qiskit import QuantumCircuit, transpile
      from qiskit.test.mock import FakeBoeblingen
      backend = FakeBoeblingen()

      ghz = QuantumCircuit(5)
      ghz.h(0)
      ghz.cx(0,range(1,5))
      ghz.draw(output='mpl')


   .. jupyter-execute::

      for kk in range(4):
         circ = transpile(ghz, backend, optimization_level=kk)
         print('Optimization Level {}'.format(kk))
         print('Depth:', circ.depth())
         print('Gate counts:', circ.count_ops())
         print()


   .. raw:: html

      <br>

.. dropdown:: Scheduling
   :animate: fade-in-slide-down

   After the circuit has been translated to the target basis, mapped to the device, and optimized,
   a scheduling phase can be applied to optionally account for all the idle time in the circuit.
   At a high level the scheduling can be thought of as inserting delays into the circuit to account
   for idle time on the qubits between the execution of instructions. For example, if we start with a
   circuit such as:

   .. jupyter-execute::

      from qiskit import QuantumCircuit, transpile
      from qiskit.test.mock import FakeBoeblingen
      backend = FakeBoeblingen()

      ghz = QuantumCircuit(5)
      ghz.h(0)
      ghz.cx(0,range(1,5))
      ghz.draw(output='mpl')

   we can then call :func:`~.transpile` on it with ``scheduling_method`` set:

   .. jupyter-execute::

      circ = transpile(ghz, backend, scheduling_method="asap")
      circ.draw(output='mpl')

   You can see here that the transpiler inserted :class:`~qiskit.circuit.Delay` instructions to
   account for idle time on each qubit. To get a better idea of the timing of the circuit we can
   also look at it with the :func:`.timeline.draw` function:

   .. jupyter-execute::

      from qiskit.visualization.timeline import draw as timeline_draw

      timeline_draw(circ)

   The scheduling of a circuit involves two parts, analysis and constraint mapping followed by a
   padding pass. The first part requires running a scheduling analysis pass such as
   :class:`~.ALAPSchedulingAnalysis` or :class:`~.ASAPSchedulingAnalysis` which analyzes the circuit
   and records the start time of each instruction in the circuit using a scheduling algorithm ("as late
   as possible" for  :class:`~.ALAPSchedulingAnalysis` and "as soon as possible" for
   :class:`~.ASAPSchedulingAnalysis`) in the property set. Once the circuit has an initial scheduling
   additional passes can be run to account for any timing constraints on the target backend, such
   as alignment constraints. This is typically done with the
   :class:`~.ConstrainedReschedule` pass which will adjust the scheduling
   set in the property set to the contraints of the target backend. Once all
   the scheduling and adjustments/rescheduling are finished a padding pass,
   such as :class:`~.PadDelay` or :class:`~.PadDynamicalDecoupling` is run
   to insert the instructions into the circuit, which completes the scheduling.

   Scheduling Anaylsis with control flow instructions:

   When scheduling analysis passes run there are additional constraints on classical conditions
   and control flow instructions in a circuit. This section covers the details of these additional
   constraints that any scheduling pass will need to account for.

   Policy of topological node ordering in scheduling:

   The DAG representation of ``QuantumCircuit`` respects the node ordering also in the
   classical register wires, though theoretically two conditional instructions
   conditioned on the same register are commute, i.e. read-access to the
   classical register doesn't change its state.

   .. parsed-literal::

       qc = QuantumCircuit(2, 1)
       qc.delay(100, 0)
       qc.x(0).c_if(0, True)
       qc.x(1).c_if(0, True)

   The scheduler SHOULD comply with above topological ordering policy of the DAG circuit.
   Accordingly, the `asap`-scheduled circuit will become

   .. parsed-literal::

            ┌────────────────┐   ┌───┐
       q_0: ┤ Delay(100[dt]) ├───┤ X ├──────────────
            ├────────────────┤   └─╥─┘      ┌───┐
       q_1: ┤ Delay(100[dt]) ├─────╫────────┤ X ├───
            └────────────────┘     ║        └─╥─┘
                              ┌────╨────┐┌────╨────┐
       c: 1/══════════════════╡ c_0=0x1 ╞╡ c_0=0x1 ╞
                              └─────────┘└─────────┘

   Note that this scheduling might be inefficient in some cases,
   because the second conditional operation can start without waiting the delay of 100 dt.
   However, such optimization should be done by another pass,
   otherwise scheduling may break topological ordering of the original circuit.

   Realistic control flow scheduling respecting for microarcitecture:

   In the dispersive QND readout scheme, qubit is measured with microwave stimulus to qubit (Q)
   followed by resonator ring-down (depopulation). This microwave signal is recorded
   in the buffer memory (B) with hardware kernel, then a discriminated (D) binary value
   is moved to the classical register (C).
   The sequence from t0 to t1 of the measure instruction interval might be modeled as follows:

   .. parsed-literal::

       Q ░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░
       B ░░▒▒▒▒▒▒▒▒░░░░░░░░░
       D ░░░░░░░░░░▒▒▒▒▒▒░░░
       C ░░░░░░░░░░░░░░░░▒▒░

   However, ``QuantumCircuit`` representation is not enough accurate to represent
   this model. In the circuit representation, thus ``Qubit`` is occupied by the
   stimulus microwave signal during the first half of the interval,
   and ``Clbit`` is only occupied at the very end of the interval.

   This precise model may induce weird edge case.

   .. parsed-literal::

               ┌───┐
       q_0: ───┤ X ├──────
               └─╥─┘   ┌─┐
       q_1: ─────╫─────┤M├
            ┌────╨────┐└╥┘
       c: 1/╡ c_0=0x1 ╞═╩═
            └─────────┘ 0

   In this example, user may intend to measure the state of ``q_1``, after ``XGate`` is
   applied to the ``q_0``. This is correct interpretation from viewpoint of
   the topological node ordering, i.e. x gate node come in front of the measure node.
   However, according to the measurement model above, the data in the register
   is unchanged during the stimulus, thus two nodes are simultaneously operated.
   If one `alap`-schedule this circuit, it may return following circuit.

   .. parsed-literal::

            ┌────────────────┐   ┌───┐
       q_0: ┤ Delay(500[dt]) ├───┤ X ├──────
            └────────────────┘   └─╥─┘   ┌─┐
       q_1: ───────────────────────╫─────┤M├
                              ┌────╨────┐└╥┘
       c: 1/══════════════════╡ c_0=0x1 ╞═╩═
                              └─────────┘ 0

   Note that there is no delay on ``q_1`` wire, and the measure instruction immediately
   start after t=0, while the conditional gate starts after the delay.
   It looks like the topological ordering between the nodes are flipped in the scheduled view.
   This behavior can be understood by considering the control flow model described above,

   .. parsed-literal::

       : Quantum Circuit, first-measure
       0 ░░░░░░░░░░░░▒▒▒▒▒▒░
       1 ░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░

       : In wire q0
       Q ░░░░░░░░░░░░░░░▒▒▒░
       C ░░░░░░░░░░░░▒▒░░░░░

       : In wire q1
       Q ░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░
       B ░░▒▒▒▒▒▒▒▒░░░░░░░░░
       D ░░░░░░░░░░▒▒▒▒▒▒░░░
       C ░░░░░░░░░░░░░░░░▒▒░

   Since there is no qubit register (Q0, Q1) overlap, the node ordering is determined by the
   shared classical register C. As you can see, the execution order is still
   preserved on C, i.e. read C then apply ``XGate``, finally store the measured outcome in C.
   Because ``DAGOpNode`` cannot define different durations for associated registers,
   the time ordering of two nodes is inverted anyways.

   This behavior can be controlled by ``clbit_write_latency`` and ``conditional_latency``.
   The former parameter determines the delay of the register write-access from
   the beginning of the measure instruction t0, and another parameter determines
   the delay of conditional gate operation from t0 which comes from the register read-access.
   These information might be found in the backend configuration and then should
   be copied to the pass manager property set before the pass is called.

   By default latencies, the `alap`-scheduled circuit of above example may become

   .. parsed-literal::

               ┌───┐
       q_0: ───┤ X ├──────
               └─╥─┘   ┌─┐
       q_1: ─────╫─────┤M├
            ┌────╨────┐└╥┘
       c: 1/╡ c_0=0x1 ╞═╩═
            └─────────┘ 0

   If the backend microarchitecture supports smart scheduling of the control flow, i.e.
   it may separately schedule qubit and classical register,
   insertion of the delay yields unnecessary longer total execution time.

   .. parsed-literal::
       : Quantum Circuit, first-xgate
       0 ░▒▒▒░░░░░░░░░░░░░░░
       1 ░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░

       : In wire q0
       Q ░▒▒▒░░░░░░░░░░░░░░░
       C ░░░░░░░░░░░░░░░░░░░ (zero latency)

       : In wire q1
       Q ░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░
       C ░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░ (zero latency, scheduled after C0 read-access)

   However this result is much more intuitive in the topological ordering view.
   If finite conditional latency is provided, for example, 30 dt, the circuit
   is scheduled as follows.

   .. parsed-literal::

            ┌───────────────┐   ┌───┐
       q_0: ┤ Delay(30[dt]) ├───┤ X ├──────
            ├───────────────┤   └─╥─┘   ┌─┐
       q_1: ┤ Delay(30[dt]) ├─────╫─────┤M├
            └───────────────┘┌────╨────┐└╥┘
       c: 1/═════════════════╡ c_0=0x1 ╞═╩═
                             └─────────┘ 0

   with the timing model:

   .. parsed-literal::
       : Quantum Circuit, first-xgate
       0 ░░▒▒▒░░░░░░░░░░░░░░░
       1 ░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░

       : In wire q0
       Q ░░▒▒▒░░░░░░░░░░░░░░░
       C ░▒░░░░░░░░░░░░░░░░░░ (30dt latency)

       : In wire q1
       Q ░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░
       C ░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░

   See https://arxiv.org/abs/2102.01682 for more details.

   .. raw:: html

      <br>


Transpiler API
==============

Transpiler Target
-----------------

.. autosummary::
   :toctree: ../stubs/

   Target
   InstructionProperties

Pass Manager Construction
-------------------------

.. autosummary::
   :toctree: ../stubs/

   PassManager
   PassManagerConfig
   PropertySet
   FlowController
   ConditionalController
   DoWhileController

Layout and Topology
-------------------

.. autosummary::
   :toctree: ../stubs/

   Layout
   CouplingMap

Scheduling
----------

.. autosummary::
   :toctree: ../stubs/

   InstructionDurations

Fenced Objects
--------------

.. autosummary::
   :toctree: ../stubs/

   FencedDAGCircuit
   FencedPropertySet

Abstract Passes
---------------

.. autosummary::
   :toctree: ../stubs/

   TransformationPass
   AnalysisPass

Exceptions
----------

.. autosummary::
   :toctree: ../stubs/

   TranspilerError
   TranspilerAccessError
"""

from .runningpassmanager import FlowController, ConditionalController, DoWhileController
from .passmanager import PassManager
from .passmanager_config import PassManagerConfig
from .propertyset import PropertySet
from .exceptions import TranspilerError, TranspilerAccessError
from .fencedobjs import FencedDAGCircuit, FencedPropertySet
from .basepasses import AnalysisPass, TransformationPass
from .coupling import CouplingMap
from .layout import Layout
from .instruction_durations import InstructionDurations
from .target import Target
from .target import InstructionProperties
from .target import QubitProperties
