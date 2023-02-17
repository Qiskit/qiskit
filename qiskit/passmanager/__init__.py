# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
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

Qiskit pass manager is somewhat inspired by the LLVM compiler,
but it is designed to take Qiskit object representation instead of plain code.

Pass manager converts the input program into an arbitrary internal representation (IR),
and it can be optimized and get lowered with a variety of transformations over multiple passes.
Finary, the IR is converted back to some Qiskit object representation.

Execution of transform passes is managed by the :class:`.FlowController`, which is an iterable
object that iterates over associated transform passes.
The flow controller may implement control flow based on the :class:`.PropertySet` representing
the state of transformation.
Each transform pass also has access to the same property set, and the pass can
obtain the status of predecessor transforms from the property set,
and it can propagate own status to the successor passes through the property set.

The user interface must be implemented with the base class :class:`BasePassManager`.
The responsibility of the pass manager is to build a sequence of flow controllers,
and to spawn a :class:`BasePassRunner` instance to run transforms.
The spawned pass runner always takes a single input program and
returns a single optimized program instance.

The pass manager supports parallelism when it receives multiple input programs.
Multiple pass runners are generated for every input program in this case,
and they run in parallel in different threads.

The responsibility of the :class:`BasePassRunner` is to convert the input program
into the pass manager IR, to run flow controllers, and finally to convert the IR back to
the Qiskit object representation. Note that the input, output and IR types are not
defined in the base class, thus one must implement a :class:`BasePassRunner` subclass
for every combination of these data types.
This "vail of ignorance" allows us to deal with the most efficient IR for a particular
optimization task, and to reuse the pass flow control machinery
for different input and output programs.


Base classes
------------

.. autosummary::
   :toctree: ../stubs/

   BasePassRunner
   BasePassManager

Flow controllers
----------------

.. autosummary::
   :toctree: ../stubs/

    FlowController
    ConditionalController
    DoWhileController

PropertySet
-----------

.. autosummary::
   :toctree: ../stubs/

   PropertySet
   FuturePropertySet

Exceptions
----------

.. autosummary::
   :toctree: ../stubs/

    PassManagerError

"""

from .base_pass_runner import BasePassRunner
from .base_passmanager import BasePassManager
from .flow_controller import FlowController, ConditionalController, DoWhileController
from .base_pass import BasePass
from .propertyset import PropertySet, FuturePropertySet
from .exceptions import PassManagerError
