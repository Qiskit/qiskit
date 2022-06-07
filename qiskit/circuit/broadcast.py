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


"""Broadcasting functionality."""

from typing import List, Tuple
from qiskit.circuit.operation import Operation
from qiskit.circuit.gate import Gate
from qiskit.circuit.exceptions import CircuitError, QiskitError


def broadcast_barrier(operation, qargs, cargs):  # pylint: disable=unused-argument
    """Broadcasting for barriers."""
    yield [qarg for sublist in qargs for qarg in sublist], []


def broadcast_measure(operation, qargs, cargs):  # pylint: disable=unused-argument
    """Broadcasting for measures."""
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


def broadcast_reset(operation, qargs, cargs):  # pylint: disable=unused-argument
    """Broadcasting for resets."""
    for qarg in qargs[0]:
        yield [qarg], []


def broadcast_delay(operation, qargs, cargs):  # pylint: disable=unused-argument
    """Broadcasting for delays."""
    yield [qarg for sublist in qargs for qarg in sublist], []


def _broadcast_single_argument_gate(qarg: List) -> List:
    """Expands a single argument.

    For example: [q[0], q[1]] -> [q[0]], [q[1]]
    """
    # [q[0], q[1]] -> [q[0]]
    #              -> [q[1]]
    for arg0 in qarg:
        yield [arg0], []


def _broadcast_2_arguments_gate(qarg0: List, qarg1: List) -> List:
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


def _broadcast_3_or_more_args_gate(qargs: List) -> List:
    if all(len(qarg) == len(qargs[0]) for qarg in qargs):
        for arg in zip(*qargs):
            yield list(arg), []
    else:
        raise CircuitError("Not sure how to combine these qubit arguments:\n %s\n" % qargs)


def broadcast_gate(operation: Gate, qargs: List, cargs: List) -> Tuple[List, List]:
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
        operation: gate being broadcasted
        qargs: List of quantum bit arguments.
        cargs: List of classical bit arguments.

    Yields:
        A tuple with single arguments.

    Raises:
        CircuitError: If the input is not valid. For example, the number of
            arguments does not match the gate expectation.
    """
    if len(qargs) != operation.num_qubits or cargs:
        raise CircuitError(
            f"The amount of qubit({len(qargs)})/clbit({len(cargs)}) arguments does"
            f" not match the gate expectation ({operation.num_qubits})."
        )

    if any(not qarg for qarg in qargs):
        raise CircuitError("One or more of the arguments are empty")

    if len(qargs) == 1:
        yield from _broadcast_single_argument_gate(qargs[0])
    elif len(qargs) == 2:
        yield from _broadcast_2_arguments_gate(qargs[0], qargs[1])
    elif len(qargs) >= 3:
        yield from _broadcast_3_or_more_args_gate(qargs)
    else:
        raise CircuitError("This gate cannot handle %i arguments" % len(qargs))


def broadcast_generic(operation: Operation, qargs, cargs):
    """
    Validation of the arguments.

    Args:
        operation: Operation to broadcast
        qargs (List): List of quantum bit arguments.
        cargs (List): List of classical bit arguments.

    Yields:
        Tuple(List, List): A tuple with single arguments.

    Raises:
        CircuitError: If the input is not valid. For example, the number of
            arguments does not match the gate expectation.
    """
    if len(qargs) != operation.num_qubits:
        raise CircuitError(
            f"The amount of qubit arguments {len(qargs)} does not match"
            f" the operation expectation ({operation.num_qubits})."
        )

    #  [[q[0], q[1]], [c[0], c[1]]] -> [q[0], c[0]], [q[1], c[1]]
    flat_qargs = [qarg for sublist in qargs for qarg in sublist]
    flat_cargs = [carg for sublist in cargs for carg in sublist]
    yield flat_qargs, flat_cargs


def broadcast_initialize(operation, qargs, cargs):
    """Broadcasting for Initializers."""
    yield from broadcast_arguments(operation._stateprep, qargs, cargs)


def broadcast_state_preparation(operation, qargs, cargs):  # pylint: disable=unused-argument
    """Broadcasting for StatePreparations."""
    flat_qargs = [qarg for sublist in qargs for qarg in sublist]

    if operation.num_qubits != len(flat_qargs):
        raise QiskitError(
            "StatePreparation parameter vector has %d elements, therefore expects %s "
            "qubits. However, %s were provided."
            % (2**operation.num_qubits, operation.num_qubits, len(flat_qargs))
        )
    yield flat_qargs, []


# Special implementations
broadcast_implementations = {
    "barrier": broadcast_barrier,
    "measure": broadcast_measure,
    "reset": broadcast_reset,
    "delay": broadcast_delay,
    "initialize": broadcast_initialize,
    "state_preparation": broadcast_state_preparation,
    "state_preparation_dg": broadcast_state_preparation,
}


def broadcast_arguments(operation: Operation, qargs, cargs):
    """The main function to broadcast arguments based on the operation."""

    # print(f"broadcast_arguments: {operation = }, {qargs = }, {cargs = }")
    if operation.name in broadcast_implementations.keys():
        # Use a custom broadcast implementation when available
        broadcaster = broadcast_implementations[operation.name]
        yield from broadcaster(operation, qargs, cargs)
    elif isinstance(operation, Gate):
        # Use implementation for gates, if the operation is a gate
        yield from broadcast_gate(operation, qargs, cargs)
    else:
        # Use the generic implementation otherwise
        yield from broadcast_generic(operation, qargs, cargs)
