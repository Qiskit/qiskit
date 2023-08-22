# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test primitives that check what kind of operations are in the circuits they execute."""

from qiskit.primitives import Estimator, Sampler


class LoggingEstimator(Estimator):
    """An estimator checking what operations were in the circuits it executed."""

    def __init__(self, options=None, operations_callback=None):
        super().__init__(options=options)
        self.operations_callback = operations_callback

    def _run(self, circuits, observables, parameter_values, **run_options):
        if self.operations_callback is not None:
            ops = [circuit.count_ops() for circuit in circuits]
            self.operations_callback(ops)
        return super()._run(circuits, observables, parameter_values, **run_options)


class LoggingSampler(Sampler):
    """A sampler checking what operations were in the circuits it executed."""

    def __init__(self, operations_callback):
        super().__init__()
        self.operations_callback = operations_callback

    def _run(self, circuits, parameter_values, **run_options):
        ops = [circuit.count_ops() for circuit in circuits]
        self.operations_callback(ops)
        return super()._run(circuits, parameter_values, **run_options)
