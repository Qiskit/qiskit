# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
=======================================
Passmanager (:mod:`qiskit.passmanager`)
=======================================

.. currentmodule:: qiskit.passmanager

Overview
========

The Qiskit pass manager is somewhat inspired by the `LLVM compiler <https://llvm.org/>`_,
but it is designed to take a Python object as an input instead of plain source code.

The pass manager converts the input Python object into an intermediate representation (IR),
and it can be optimized and get lowered with a variety of transformations over multiple passes.
The pass manager framework may employ multiple IRs with interleaved conversion passes,
depending on the context of the compilation. Finally, the IR is converted back to some Python object.
Note that the input type and output type are not necessarily the same.

Compilation in the pass manager is a chain of :class:`~.passmanager.Task` executions that
take an IR and output a new IR with some optimization or data analysis.
An atomic task is a *pass* which is a subclass of :class:`.GenericPass` that implements
a :meth:`.~GenericPass.run` method that performs some work on the received IR.
A set of passes may form a *flow controller*, which is a subclass of
:class:`.BaseController`, which can implement arbitrary compilation-state-dependent logic for
deciding which pass will get run next.
Passes share intermediate data via the :class:`.PropertySet` object which is
a free-form dictionary. A pass can populate the property set dictionary during the task execution.
A flow controller can also consume the property set to control the pass execution,
but this access must be read-only.
The property set is portable and handed over from pass to pass at execution.
In addition to the property set, tasks also receive a :class:`.WorkflowStatus` data structure.
This object is initialized when the pass manager is run and handed over to underlying tasks.
The status is updated after every pass is run, and contains information about the pipeline state
(number of passes run, failure state, and so on) as opposed to the :class:`PropertySet`, which
contains information about the IR being optimized.

A :class:`BasePassManager` is a wrapper of the flow controller, with responsibilities of

* Scheduling tasks,
* Converting an input Python object to an internal IR,
* Initializing a property set and workflow status,
* Running scheduled tasks to apply a series of transformations to the IR,
* Converting the IR back to an output Python object.

This indicates that the flow controller itself is type-agnostic, and a developer must
implement a subclass of the :class:`BasePassManager` to manage the data conversion steps.
This *veil of ignorance* allows us to choose the most efficient data representation
for a particular pass manager task, while we can reuse the flow control machinery
for different input and output types.

A single flow controller always takes a single IR object, and returns a single
IR object. Parallelism for multiple input objects is supported by the
:class:`BasePassManager` by broadcasting the flow controller via
the :func:`.parallel_map` function.

The :class:`MultiStagePassManager` allows constructing staged compiler workflows with
multiple IRs. A stage is defined by a :class:`~.passmanager.Task` or an iterable thereof, which can
also be grouped inside a :class:`BasePassManager`.
The stages must be set up such that the output IR of the current stage matches the input IR
of the next stage, there are (currently) no automatic translations.

Multi-IR staged workflows
-------------------------

:class:`MultiStagePassManager` chains named compilation stages where each stage can
preserve or lower the intermediate representation. Unlike
:class:`~qiskit.transpiler.StagedPassManager`, which is specialized for
:class:`~.DAGCircuit` pipelines with implicit :class:`~.QuantumCircuit`
conversion, :class:`MultiStagePassManager` accepts an arbitrary sequence of IR
types as long as each stage's output matches the next stage's input.

A stage may be:

* a single :class:`.GenericPass` (or other :class:`~.passmanager.Task`) instance,
* an iterable of tasks executed in order, or
* a :class:`BasePassManager` used as a *task container*. When a pass manager is
  provided as a stage, only its scheduled tasks are executed; the stage does not
  invoke :meth:`~.BasePassManager._passmanager_frontend` or
  :meth:`~.BasePassManager._passmanager_backend`.

.. note::

    It is the user's responsibility to connect compatible stages. The pass
    manager does not insert automatic IR conversion passes between stages.

When embedding existing transpiler pipelines, a pass manager returned by
:func:`~qiskit.transpiler.generate_preset_pass_manager` may be used directly as
a DAG stage.

The :class:`~.passmanager.Task` interface is internal. Custom compilation steps
should subclass :class:`.GenericPass` and declare input and output IR types via
type parameters, for example ``GenericPass[PauliIR, QuantumCircuit]``.


Examples
========

We look into a toy optimization task, namely, preparing a row of numbers
and removing a digit if the number is five.
Such a task might be easily done by converting the input numbers into string.
We use the pass manager framework here, putting the efficiency aside for
a moment to learn how to build a custom Qiskit compiler.

.. plot::
   :include-source:
   :nofigs:
   :context:

    from qiskit.passmanager import BasePassManager, GenericPass, ConditionalController

    class ToyPassManager(BasePassManager):

        def _passmanager_frontend(self, input_program: int, **kwargs) -> str:
            return str(input_program)

        def _passmanager_backend(self, passmanager_ir: str, in_program: int, **kwargs) -> int:
            return int(passmanager_ir)

This pass manager inputs and outputs an integer number, while
performing the optimization tasks on a string data.
Hence, input, IR, output type are integer, string, integer, respectively.
The :meth:`.~BasePassManager._passmanager_frontend` method defines the conversion from the
input data to IR, and :meth:`.~BasePassManager._passmanager_backend` defines
the conversion from the IR to output data.
The pass manager backend is also given an :code:`in_program` parameter that contains the original
``input_program`` to the front end, for referencing any original metadata of the input program for
the final conversion.

Next, we implement a pass that removes a digit when the number is five.

.. plot::
   :include-source:
   :nofigs:
   :context:

    class RemoveFive(GenericPass):

        def run(self, passmanager_ir: str):
            return passmanager_ir.replace("5", "")

    task = RemoveFive()

Finally, we instantiate a pass manager and schedule the task with it.
Running the pass manager with a random row of numbers returns
new numbers that don't contain five.

.. plot::
   :include-source:
   :nofigs:
   :context:

    pm = ToyPassManager()
    pm.append(task)

    pm.run([123456789, 45654, 36785554])

Output:

.. code-block:: text

    [12346789, 464, 36784]

Now we consider the case of conditional execution.
We avoid execution of the "remove five" task when the input number is
six digits or less. Such control can be implemented by a flow controller.
We start from an analysis pass that provides the flow controller
with information about the number of digits.

.. plot::
   :include-source:
   :nofigs:
   :context:

    class CountDigits(GenericPass):

        def run(self, passmanager_ir: str):
            self.property_set["ndigits"] = len(passmanager_ir)

    analysis_task = CountDigits()

Then, we wrap the remove five task with the :class:`.ConditionalController`
that runs the stored tasks only when the condition is met.

.. plot::
   :include-source:
   :nofigs:
   :context:

    def digit_condition(property_set):
        # Return True when condition is met.
        return property_set["ndigits"] > 6

    conditional_task = ConditionalController(
        tasks=[RemoveFive()],
        condition=digit_condition,
    )

As before, we schedule these passes with the pass manager and run.

.. plot::
   :include-source:
   :nofigs:
   :context:

    pm = ToyPassManager()
    pm.append(analysis_task)
    pm.append(conditional_task)

    pm.run([123456789, 45654, 36785554])

Output:

.. code-block:: text

    [12346789, 45654, 36784]

The "remove five" task is triggered only for the first and third input
values, which have more than six digits.

With the pass manager framework, a developer can flexibly customize
the optimization task by combining multiple passes and flow controllers.
See details in the following class API documentation.

Multi-IR staged pipelines
-------------------------

The following example shows a minimal multi-IR pipeline. A custom Pauli-string
IR is optimized, lowered to a :class:`~.QuantumCircuit`, decomposed, and
finally converted to a :class:`~.DAGCircuit`.

.. plot::
   :include-source:
   :nofigs:
   :context:

    from qiskit.circuit import QuantumCircuit
    from qiskit.converters import circuit_to_dag
    from qiskit.dagcircuit import DAGCircuit
    from qiskit.passmanager import BasePassManager, GenericPass, MultiStagePassManager

    class PauliIR:
        """A toy IR storing global Pauli strings."""

        def __init__(self, num_qubits: int):
            self.num_qubits = num_qubits
            self.instructions = []

        def apply(self, pauli: str):
            if len(pauli) != self.num_qubits:
                raise ValueError("Incompatible number of qubits")
            self.instructions.append(pauli)

    class PauliPM(BasePassManager):
        """A pass manager used only as a task container for the Pauli IR."""

        def _passmanager_frontend(self, input_program, **kwargs):
            return input_program

        def _passmanager_backend(self, passmanager_ir, in_program, **kwargs):
            return passmanager_ir

    class RemovePauliIdentities(GenericPass[PauliIR, PauliIR]):
        """Remove all-identity Pauli strings from the IR."""

        def run(self, passmanager_ir: PauliIR) -> PauliIR:
            passmanager_ir.instructions = [
                pauli for pauli in passmanager_ir.instructions if any(p != "I" for p in pauli)
            ]
            return passmanager_ir

    class PauliToCircuit(GenericPass[PauliIR, QuantumCircuit]):
        """Lower PauliIR to QuantumCircuit."""

        def run(self, passmanager_ir: PauliIR) -> QuantumCircuit:
            circuit = QuantumCircuit(passmanager_ir.num_qubits)
            for pauli in passmanager_ir.instructions:
                circuit.pauli(pauli, circuit.qubits)
            return circuit

    class CircuitDecomposer(GenericPass[QuantumCircuit, QuantumCircuit]):
        """Decompose the circuit one level."""

        def run(self, passmanager_ir: QuantumCircuit) -> QuantumCircuit:
            return passmanager_ir.decompose()

    class CircuitToDAG(GenericPass[QuantumCircuit, DAGCircuit]):
        """Lower QuantumCircuit to DAGCircuit."""

        def run(self, passmanager_ir: QuantumCircuit) -> DAGCircuit:
            return circuit_to_dag(passmanager_ir)

    pauli_pm = PauliPM([RemovePauliIdentities()])
    multi_pm = MultiStagePassManager(
        pauli=pauli_pm,
        pauli_to_circuit=PauliToCircuit(),
        circuit=CircuitDecomposer(),
        circuit_to_dag=CircuitToDAG(),
    )

    program = PauliIR(3)
    program.apply("XYZ")
    program.apply("III")
    program.apply("ZZI")

    output_dag = multi_pm.run(program)

An existing transpiler pipeline can be embedded as a stage once the program has
been lowered to a :class:`~.DAGCircuit`:

.. plot::
   :include-source:
   :nofigs:
   :context:

    from qiskit.transpiler import CouplingMap, generate_preset_pass_manager

    dag_pm = generate_preset_pass_manager(
        coupling_map=CouplingMap.from_line(3), basis_gates=["sx", "x", "rz", "cx"]
    )

    transpile_pm = MultiStagePassManager(
        pauli=pauli_pm,
        pauli_to_circuit=PauliToCircuit(),
        circuit_to_dag=CircuitToDAG(),
        dag=dag_pm,
    )

    transpiled = transpile_pm.run(program)

Stages are fixed at construction time, but their implementations can be replaced
by assigning to the corresponding attribute:

.. plot::
   :include-source:
   :nofigs:
   :context:

    from qiskit.transpiler.passes import RemoveIdentityEquivalent

    multi_pm.circuit_to_dag = CircuitToDAG()
    multi_pm.dag_opt = RemoveIdentityEquivalent()


Interface
=========

Base classes
------------

.. autosummary::
   :toctree: ../stubs/

   BasePassManager
   BaseController
   GenericPass


Pass managers
-------------

.. autosummary::
   :toctree: ../stubs/

   MultiStagePassManager

Flow controllers
----------------

.. autosummary::
   :toctree: ../stubs/

   FlowControllerLinear
   ConditionalController
   DoWhileController

Compilation state
-----------------

.. autosummary::
   :toctree: ../stubs/

   PropertySet
   WorkflowStatus
   PassManagerState

Exceptions
----------

.. autoexception:: PassManagerError
"""

from .passmanager import BasePassManager
from .multistage_passmanager import MultiStagePassManager
from .flow_controllers import (
    FlowControllerLinear,
    ConditionalController,
    DoWhileController,
)
from .base_tasks import GenericPass, BaseController
from .compilation_status import PropertySet, WorkflowStatus, PassManagerState
from .exceptions import PassManagerError

__all__ = [
    "BaseController",
    "BasePassManager",
    "ConditionalController",
    "DoWhileController",
    "FlowControllerLinear",
    "GenericPass",
    "MultiStagePassManager",
    "PassManagerError",
    "PassManagerState",
    "PropertySet",
    "WorkflowStatus",
]
