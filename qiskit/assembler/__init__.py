# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
========================================================
Circuit and Schedule Assembler (:mod:`qiskit.assembler`)
========================================================

.. currentmodule:: qiskit.assembler

Circuit Assembler
=================

.. autosummary::
   :toctree: ../stubs/

   assemble_circuits
   ===============
   .. code-block::
      from qiskit import BasicAer
      from qiskit.circuit import QuantumRegister, ClassicalRegister, QuantumCircuit
      from qiskit.assembler import assemble_circuits
      from qiskit.assembler.run_config import RunConfig
      sim_backend = BasicAer.get_backend("qasm_simulator")
      q = QuantumRegister(2)
      c = ClassicalRegister(2)
      qc = QuantumCircuit(q, c)
      qc.h(q[0])
      qc.cx(q[0], q[1])
      qc.measure(q, c)
      qobj = assemble_circuits(circuits=[qc],
                              qobj_id="custom-id",
                              qobj_header=[],
                              run_config=RunConfig(shots=2000, memory=True, init_qubits=True))
      print(qobj) # output QASM Qobj

Schedule Assembler
==================

.. autosummary::
   :toctree: ../stubs/

   assemble_schedules
   ===============
   .. code-block::
      from qiskit import pulse
      from qiskit.compiler.assembler import assemble
      from qiskit.assembler import assemble_schedules
      from qiskit.assembler.run_config import RunConfig

      header = {"backend_name": "FakeOpenPulse2Q", "backend_version": "0.0.0"}
      config = RunConfig(shots=1024,
                           memory=False,
                           meas_level=1,
                           meas_return='avg',
                           memory_slot_size=100,
                           parametric_pulses=[],
                           init_qubits=True,
                           qubit_lo_freq=[4900000000.0, 5000000000.0],
                           meas_lo_freq=[6500000000.0, 6600000000.0],
                           schedule_los=[])
      default_qubit_lo_freq = [4.9e9, 5.0e9]
      default_meas_lo_freq = [6.5e9, 6.6e9]
      schedule = pulse.Schedule()
      schedule += pulse.Play(
         pulse.Waveform([0.1] * 16, name="test0"), pulse.DriveChannel(0), name="test1"
      )
      schedule += pulse.Play(
         pulse.Waveform([0.1] * 16, name="test1"), pulse.DriveChannel(0), name="test2"
      )
      schedule += pulse.Play(
         pulse.Waveform([0.5] * 16, name="test0"), pulse.DriveChannel(0), name="test1"
      )
      pulseQobj = assemble_schedules(schedules=[schedule],
                                       qobj_id="custom-id",
                                       qobj_header=header,
                                       run_config=config)

Disassembler
============

.. autosummary::
   :toctree: ../stubs/

   disassemble
   ===============
   .. code-block::

      from qiskit import BasicAer
      from qiskit.circuit import QuantumRegister, ClassicalRegister, QuantumCircuit
      from qiskit.compiler.assembler import assemble
      from qiskit.assembler.disassemble import disassemble
      sim_backend = BasicAer.get_backend("qasm_simulator")
      q = QuantumRegister(2)
      c = ClassicalRegister(2)
      qc = QuantumCircuit(q, c)
      qc.h(q[0])
      qc.cx(q[0], q[1])
      qc.measure(q, c)
      qobj = assemble(qc, shots=2000, memory=True)
      circuits, run_config_out, headers = disassemble(qobj)
      print(circuits[0].draw())

RunConfig
=========

.. autosummary::
   :toctree: ../stubs/

   RunConfig
"""

from qiskit.assembler.assemble_circuits import assemble_circuits
from qiskit.assembler.assemble_schedules import assemble_schedules
from qiskit.assembler.disassemble import disassemble
from qiskit.assembler.run_config import RunConfig
