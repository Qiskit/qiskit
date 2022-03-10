# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Arguments Broadcaster Mixin."""

from abc import ABC, abstractmethod
from typing import List, Tuple
from qiskit.circuit.exceptions import CircuitError, QiskitError


class ArgumentsBroadcaster(ABC):
    """
    A mixin for argument broadcasting.
    """

    @abstractmethod
    def broadcast_arguments(self, qargs, cargs):
        pass


class ArgumentsBroadcasterGeneric(ArgumentsBroadcaster):
    def broadcast_arguments(self, qargs, cargs):
        """
        Validation of the arguments.

        Args:
            qargs (List): List of quantum bit arguments.
            cargs (List): List of classical bit arguments.

        Yields:
            Tuple(List, List): A tuple with single arguments.

        Raises:
            CircuitError: If the input is not valid. For example, the number of
                arguments does not match the gate expectation.
        """
        # This is the "generic" method, formerly implemented by Instruction class.
        #print("In AB: generic")
        if len(qargs) != self.num_qubits:
            raise CircuitError(
                f"The amount of qubit arguments {len(qargs)} does not match"
                f" the instruction expectation ({self.num_qubits})."
            )

        #  [[q[0], q[1]], [c[0], c[1]]] -> [q[0], c[0]], [q[1], c[1]]
        flat_qargs = [qarg for sublist in qargs for qarg in sublist]
        flat_cargs = [carg for sublist in cargs for carg in sublist]

        yield flat_qargs, flat_cargs


class ArgumentsBroadcasterBarrier(ArgumentsBroadcaster):

    def broadcast_arguments(self, qargs, cargs):
        #print("In AB: barrier")

        yield [qarg for sublist in qargs for qarg in sublist], []


class ArgumentsBroadcasterDelay(ArgumentsBroadcaster):

    def broadcast_arguments(self, qargs, cargs):
        #print("In AB: delay")
        yield [qarg for sublist in qargs for qarg in sublist], []


class ArgumentsBroadcasterGate(ArgumentsBroadcaster):

    @staticmethod
    def _broadcast_single_argument(qarg: List) -> List:
        """Expands a single argument.

        For example: [q[0], q[1]] -> [q[0]], [q[1]]
        """
        # [q[0], q[1]] -> [q[0]]
        #              -> [q[1]]
        for arg0 in qarg:
            yield [arg0], []

    @staticmethod
    def _broadcast_2_arguments(qarg0: List, qarg1: List) -> List:
        if len(qarg0) == len(qarg1):
            # [[q[0], q[1]], [r[0], r[1]]] -> [q[0], r[0]]
            #                              -> [q[1], r[1]]
            for arg0, arg1 in zip(qarg0, qarg1):
                yield [arg0, arg1], []
        elif len(qarg0) == 1:
            # [[q[0]], [r[0], r[1]]] -> [q[0], r[0]]
            #                        -> [q[0], r[1]]
            for arg1 in qarg1:
                yield [qarg0[0], arg1], []
        elif len(qarg1) == 1:
            # [[q[0], q[1]], [r[0]]] -> [q[0], r[0]]
            #                        -> [q[1], r[0]]
            for arg0 in qarg0:
                yield [arg0, qarg1[0]], []
        else:
            raise CircuitError(
                f"Not sure how to combine these two-qubit arguments:\n {qarg0}\n {qarg1}"
            )

    @staticmethod
    def _broadcast_3_or_more_args(qargs: List) -> List:
        if all(len(qarg) == len(qargs[0]) for qarg in qargs):
            for arg in zip(*qargs):
                yield list(arg), []
        else:
            raise CircuitError("Not sure how to combine these qubit arguments:\n %s\n" % qargs)

    def broadcast_arguments(self, qargs: List, cargs: List) -> Tuple[List, List]:
        """Validation and handling of the arguments and its relationship.

        For example, ``cx([q[0],q[1]], q[2])`` means ``cx(q[0], q[2]); cx(q[1], q[2])``. This
        method yields the arguments in the right grouping. In the given example::

            in: [[q[0],q[1]], q[2]],[]
            outs: [q[0], q[2]], []
                  [q[1], q[2]], []

        The general broadcasting rules are:

            * If len(qargs) == 1::

                [q[0], q[1]] -> [q[0]],[q[1]]

            * If len(qargs) == 2::

                [[q[0], q[1]], [r[0], r[1]]] -> [q[0], r[0]], [q[1], r[1]]
                [[q[0]], [r[0], r[1]]]       -> [q[0], r[0]], [q[0], r[1]]
                [[q[0], q[1]], [r[0]]]       -> [q[0], r[0]], [q[1], r[0]]

            * If len(qargs) >= 3::

                [q[0], q[1]], [r[0], r[1]],  ...] -> [q[0], r[0], ...], [q[1], r[1], ...]

        Args:
            qargs: List of quantum bit arguments.
            cargs: List of classical bit arguments.

        Returns:
            A tuple with single arguments.

        Raises:
            CircuitError: If the input is not valid. For example, the number of
                arguments does not match the gate expectation.
        """

        #print("In AB: gate")

        if len(qargs) != self.num_qubits or cargs:
            raise CircuitError(
                f"The amount of qubit({len(qargs)})/clbit({len(cargs)}) arguments does"
                f" not match the gate expectation ({self.num_qubits})."
            )

        if any(not qarg for qarg in qargs):
            raise CircuitError("One or more of the arguments are empty")

        if len(qargs) == 1:
            return self._broadcast_single_argument(qargs[0])
        elif len(qargs) == 2:
            return self._broadcast_2_arguments(qargs[0], qargs[1])
        elif len(qargs) >= 3:
            return self._broadcast_3_or_more_args(qargs)
        else:
            raise CircuitError("This gate cannot handle %i arguments" % len(qargs))


class ArgumentsBroadcasterMeasure(ArgumentsBroadcaster):

    def broadcast_arguments(self, qargs, cargs):

        #print("In AB: measure")

        qarg = qargs[0]
        carg = cargs[0]

        if len(carg) == len(qarg):
            for qarg, carg in zip(qarg, carg):
                yield [qarg], [carg]
        elif len(qarg) == 1 and carg:
            for each_carg in carg:
                yield qarg, [each_carg]
        else:
            raise CircuitError("register size error")


class ArgumentsBroadcasterReset(ArgumentsBroadcaster):
    def broadcast_arguments(self, qargs, cargs):
        #print("In AB: reset")

        for qarg in qargs[0]:
            yield [qarg], []


class ArgumentsBroadcasterInitializer(ArgumentsBroadcaster):
    def broadcast_arguments(self, qargs, cargs):
        #print("In AB: initializer")

        flat_qargs = [qarg for sublist in qargs for qarg in sublist]

        if self.num_qubits != len(flat_qargs):
            raise QiskitError(
                "Initialize parameter vector has %d elements, therefore expects %s "
                "qubits. However, %s were provided."
                % (2 ** self.num_qubits, self.num_qubits, len(flat_qargs))
            )
        yield flat_qargs, []

