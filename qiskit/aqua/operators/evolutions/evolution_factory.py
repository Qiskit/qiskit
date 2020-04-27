# -*- coding: utf-8 -*-

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

""" EvolutionFactory Class """

import logging

from ..operator_base import OperatorBase
from .evolution_base import EvolutionBase
from .pauli_trotter_evolution import PauliTrotterEvolution
from .matrix_evolution import MatrixEvolution

logger = logging.getLogger(__name__)


class EvolutionFactory:
    """ A factory class for convenient automatic selection of an Evolution algorithm based on the
    Operator to be converted.
    """

    @staticmethod
    def build(operator: OperatorBase = None) -> EvolutionBase:
        r"""
        A factory method for convenient automatic selection of an Evolution algorithm based on the
        Operator to be converted.

        Args:
            operator: the Operator being evolved

        Returns:
            EvolutionBase: the ``EvolutionBase`` best suited to evolve operator.

        Raises:
            ValueError: If operator is not of a composition for which we know the best Evolution
                method.

        """
        primitives = operator.primitive_strings()
        if 'Matrix' in primitives:
            return MatrixEvolution()

        elif 'Pauli' in primitives:
            # TODO figure out what to do based on qubits and hamming weight.
            return PauliTrotterEvolution()

        else:
            raise ValueError('Evolutions of mixed Operators not yet supported.')
