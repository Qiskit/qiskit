# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" TrotterizationFactory Class """

import logging

from .trotterization_base import TrotterizationBase
from .trotter import Trotter
from .suzuki import Suzuki
from .qdrift import QDrift

logger = logging.getLogger(__name__)


class TrotterizationFactory():
    """ A factory for conveniently creating TrotterizationBase instances. """

    @staticmethod
    def build(mode: str = 'trotter',
              reps: int = 1) -> TrotterizationBase:
        """ A factory for conveniently creating TrotterizationBase instances.

        Args:
            mode: One of 'trotter', 'suzuki', 'qdrift'
            reps: The number of times to repeat the Trotterization circuit.

        Returns:
            The desired TrotterizationBase instance.

        Raises:
            ValueError: A string not in ['trotter', 'suzuki', 'qdrift'] is given for mode.
        """
        if mode == 'trotter':
            return Trotter(reps=reps)

        elif mode == 'suzuki':
            return Suzuki(reps=reps)

        elif mode == 'qdrift':
            return QDrift(reps=reps)

        raise ValueError('Trotter mode {} not supported'.format(mode))
