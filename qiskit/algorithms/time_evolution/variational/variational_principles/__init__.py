# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Principles (:mod:`qiskit.algorithms.time_evolution.variational.variational_principles`)
=====================================================

It  contains a variety of variational variational_principles which can be used to simulate quantum
time evolution by propagating the parameters of a parameterized quantum circuit.

These variational variational_principles can be divided into two categories:
`Variational Quantum Imaginary Time Evolution`_
  Given a Hamiltonian, a time and a variational ansatz, the variational principle describes a
  variational principle according to the normalized Wick-rotated Schroedinger equation.
`Variational Quantum Real Time Evolution`_
  Given a Hamiltonian, a time and a variational ansatz, the variational principle describes a
  variational principle according to the Schroedinger equation.

.. currentmodule:: qiskit.algorithms.time_evolution.variational.variational_principles

Variational Principle Base Classes
====================
.. autosummary::
   :toctree: ../stubs/
   :template: autosummary/class_no_inherited_members.rst

    VariationalPrinciple
    ImaginaryVariationalPrinciple
    RealVariationalPrinciple

Variational Quantum Imaginary Time Evolution Principles
================
.. autosummary::
   :toctree: ../stubs/
   :template: autosummary/class_no_inherited_members.rst

    ImaginaryMcLachlanVariationalPrinciple

Variational Quantum Real Time Evolution Principles
=================
.. autosummary::
   :toctree: ../stubs/
   :template: autosummary/class_no_inherited_members.rst

    RealMcLachlanVariationalPrinciple
    RealTimeDependentVariationalPrinciple
"""
