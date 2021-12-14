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
Variational Quantum Time Evolutions (:mod:`qiskit.algorithms.quantum_time_evolution.variational`)
=====================================================

Algorithms for performing Variational Quantum Time Evolution of quantum states and their
gradients which might be suitable for NISQ devices. VarQte base class exposes an interface,
compliant with the Quantum Time Evolution Framework in Qiskit Terra, that is implemented by
VarQrte and VarQite classes for real and imaginary time evolution respectively. The variational
approach is taken according to a variational principle chosen by a user.

**Examples**

.. code-block::

    # define a Hamiltonian
    observable = SummedOp(
        [
            0.2252 * (I ^ I),
            0.5716 * (Z ^ Z),
            0.3435 * (I ^ Z),
            -0.4347 * (Z ^ I),
            0.091 * (Y ^ Y),
            0.091 * (X ^ X),
        ]
    ).reduce()

    # define a parametrized initial state to be evolved
    d = 1
    ansatz = EfficientSU2(observable.num_qubits, reps=d)
    parameters = ansatz.ordered_parameters

    # define values of initial parameters
    init_param_values = np.zeros(len(ansatz.ordered_parameters))
    for i in range(len(ansatz.ordered_parameters)):
        init_param_values[i] = np.pi / 2
    param_dict = dict(zip(parameters, init_param_values))

    # define a variational principle
    var_principle = ImaginaryMcLachlanVariationalPrinciple()

    # optionally define a backend
    backend = Aer.get_backend("statevector_simulator")

    # instantiate the algorithm
    var_qite = VarQite(
        var_principle, backend=backend, error_based_ode=False
    )

    # define evolution time
    time = 1

    # run the algorithm/evolve the state
    evolution_result = var_qite.evolve(
        observable,
        time,
        ansatz,
        hamiltonian_value_dict=param_dict,
    )

.. currentmodule:: qiskit.algorithms.quantum_time_evolution.variational

VarQite
--------------------

Algorithm that performs Variational Quantum Imaginary Time Evolution.

.. autosummary::
   :toctree: ../stubs/
   :template: autosummary/class_no_inherited_members.rst

   VarQite

VarQrte
--------------------

Algorithm that performs Variational Quantum Real Time Evolution.

.. autosummary::
   :toctree: ../stubs/
   :template: autosummary/class_no_inherited_members.rst

   VarQrte

VariationalPrinciples
----------

.. autosummary::
   :toctree: ../stubs/
   :template: autosummary/class_no_inherited_members.rst

    RealMcLachlanVariationalPrinciple
    RealTimeDependentVariationalPrinciple
    ImaginaryMcLachlanVariationalPrinciple
"""

from qiskit.algorithms.quantum_time_evolution.variational.variational_principles.imaginary.implementations.imaginary_mc_lachlan_variational_principle import (
    ImaginaryMcLachlanVariationalPrinciple,
)
from qiskit.algorithms.quantum_time_evolution.variational.variational_principles.real.implementations.real_mc_lachlan_variational_principle import (
    RealMcLachlanVariationalPrinciple,
)
from qiskit.algorithms.quantum_time_evolution.variational.variational_principles.real.implementations.real_time_dependent_variational_principle import (
    RealTimeDependentVariationalPrinciple,
)
from qiskit.algorithms.quantum_time_evolution.variational.var_qite import VarQite
from qiskit.algorithms.quantum_time_evolution.variational.var_qrte import VarQrte

__all__ = [
    "VarQite",
    "VarQrte",
    "RealMcLachlanVariationalPrinciple",
    "RealTimeDependentVariationalPrinciple",
    "ImaginaryMcLachlanVariationalPrinciple",
]
