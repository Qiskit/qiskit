# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
====================================================================
BasicAer: Python-based Simulators (:mod:`qiskit.providers.basicaer`)
====================================================================

.. currentmodule:: qiskit.providers.basicaer

.. deprecated:: 0.46.0

    The :mod:`qiskit.providers.basicaer` module is deprecated as of Qiskit 0.46 and will
    be removed in Qiskit 1.0. Its functionality has been replaced by the new
    :mod:`qiskit.providers.basic_provider` module and the :mod:`qiskit.quantum_info`
    module.

    The migration from using
    ``BasicAer`` to ``BasicProvider`` can be performed as follows::

        Migrate from                     |   Replace with
        ------------------------------------------------------------------------------
        :mod:`.basicaer`                 |  :mod:`.basic_provider`
        :class:`.BasicAer`               |  :class:`.BasicProvider` (*)
        :class:`.BasicAerProvider`       |  :class:`.BasicProvider` (*)
        :class:`.BasicAerJob`            |  :class:`.BasicProviderJob`
        :class:`.QasmSimulatorPy`        |  :class:`.BasicSimulator`
        :class:`.UnitarySimulatorPy`     |  use :class:`~.quantum_info.Operator`
        :class:`.StatevectorSimulatorPy` |  use :class:`~.quantum_info.Statevector`

        * The ``BasicAer`` alias maps to ``BasicProvider`` and the
          ``BasicAerProvider()`` instance maps to ``BasicProvider()``.

    This example shows the migration path of the three simulators
    in :mod:`.basicaer`::

        from qiskit import QuantumCircuit
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.h(1)
        qc.cx(1,2)
        qc.measure_all()

        # Former path
        # --------
        # 1. import BasicAer (provider)
        from qiskit import BasicAer

        # 2. run with StatevectorSimulatorPy
        backend = BasicAer.get_backend("statevector_simulator")
        statevector = backend.run(qc).result().get_statevector()

        # 3. run with UnitarySimulatorPy
        backend = BasicAer.get_backend("unitary_simulator")
        result = backend.run(qc).result()

        # 4. run with QasmSimulatorPy
        backend = BasicAer.get_backend("qasm_simulator")
        result = backend.run(qc).result()

        # New path
        # --------
        # 1. import BasicProvider
        from qiskit import BasicProvider

        # 2. run with quantum_info.Statevector
        # Note: no measurements allowed
        qc.remove_final_measurements()
        from qiskit.quantum_info import Statevector
        statevector = Statevector(qc)

        # 3. run with quantum_info.Operator
        # Note: no measurements allowed
        from qiskit.quantum_info import Operator
        result = Operator(qc).data

        # 4. run with BasicSimulator
        # Note: measurements required
        qc.measure_all()
        backend = BasicProvider.get_backend("basic_simulator")
        result = backend.run(qc).result()

A module of Python-based quantum simulators.  Simulators are accessed
via the `BasicAer` provider, e.g.:

.. code-block::

   from qiskit import BasicAer

   backend = BasicAer.get_backend('qasm_simulator')


Simulators
==========

.. autosummary::
   :toctree: ../stubs/

   QasmSimulatorPy
   StatevectorSimulatorPy
   UnitarySimulatorPy

Provider
========

.. autosummary::
   :toctree: ../stubs/

   BasicAerProvider

Job Class
=========

.. autosummary::
   :toctree: ../stubs/

   BasicAerJob

Exceptions
==========

.. autosummary::
   :toctree: ../stubs/

   BasicAerError
"""

import warnings

from .basicaerprovider import BasicAerProvider
from .basicaerjob import BasicAerJob
from .qasm_simulator import QasmSimulatorPy
from .statevector_simulator import StatevectorSimulatorPy
from .unitary_simulator import UnitarySimulatorPy
from .exceptions import BasicAerError

# Global instance to be used as the entry point for convenience.
BasicAer = BasicAerProvider()
