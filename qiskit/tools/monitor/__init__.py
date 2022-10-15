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

"""A module for monitoring jobs, backends, etc.


job_monitor
===============

.. code-block::

    from qiskit import BasicAer, execute
    from qiskit.circuit import QuantumRegister, ClassicalRegister, QuantumCircuit
    from qiskit.tools.monitor import job_monitor
    sim_backend = BasicAer.get_backend("qasm_simulator")

    q = QuantumRegister(2)
    c = ClassicalRegister(2)
    qc = QuantumCircuit(q, c)

    qc.h(q[0])
    qc.cx(q[0], q[1])
    qc.measure(q, c)
    job_sim = execute(qc, backend=sim_backend)
    job_monitor(job_sim)

"""

from .job_monitor import job_monitor
from .overview import backend_monitor, backend_overview
