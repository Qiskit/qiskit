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

"""Set default batch sizes for the optimizers."""

from qiskit.algorithms.optimizers import Optimizer, SPSA


def _set_default_batchsize(optimizer: Optimizer) -> bool:
    """Set the default batchsize, if None is set and return whether it was updated or not."""
    if isinstance(optimizer, SPSA):
        updated = optimizer._max_evals_grouped is None
        if updated:
            optimizer.set_max_evals_grouped(50)
    else:  # we only set a batchsize for SPSA
        updated = False

    return updated
