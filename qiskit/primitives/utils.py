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
"""
Utility functions for primitives
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator
from functools import lru_cache, wraps
from inspect import signature
from typing import TypeVar
from weakref import ref

from numpy.random import Generator, default_rng
from qiskit.circuit import ParameterExpression, QuantumCircuit
from qiskit.extensions.quantum_initializer.initializer import Initialize
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.symplectic.base_pauli import BasePauli


def init_circuit(state: QuantumCircuit | Statevector) -> QuantumCircuit:
    """Initialize state by converting the input to a quantum circuit.

    Args:
        state: The state as quantum circuit or statevector.

    Returns:
        The state as quantum circuit.
    """
    if isinstance(state, QuantumCircuit):
        return state
    if not isinstance(state, Statevector):
        state = Statevector(state)
    qc = QuantumCircuit(state.num_qubits)
    qc.append(Initialize(state), qargs=range(state.num_qubits))
    return qc


def init_observable(observable: BaseOperator | PauliSumOp) -> SparsePauliOp:
    """Initialize observable by converting the input to a :class:`~qiskit.quantum_info.SparsePauliOp`.

    Args:
        observable: The observable.

    Returns:
        The observable as :class:`~qiskit.quantum_info.SparsePauliOp`.

    Raises:
        TypeError: If the observable is a :class:`~qiskit.opflow.PauliSumOp` and has a parameterized
            coefficient.
    """
    if isinstance(observable, SparsePauliOp):
        return observable
    elif isinstance(observable, PauliSumOp):
        if isinstance(observable.coeff, ParameterExpression):
            raise TypeError(
                f"Observable must have numerical coefficient, not {type(observable.coeff)}."
            )
        return observable.coeff * observable.primitive
    elif isinstance(observable, BasePauli):
        return SparsePauliOp(observable)
    elif isinstance(observable, BaseOperator):
        return SparsePauliOp.from_operator(observable)
    else:
        return SparsePauliOp(observable)


def final_measurement_mapping(circuit: QuantumCircuit) -> dict[int, int]:
    """Return the final measurement mapping for the circuit.

    Dict keys label measured qubits, whereas the values indicate the
    classical bit onto which that qubits measurement result is stored.

    Note: this function is a slightly simplified version of a utility function
    ``_final_measurement_mapping`` of
    `mthree <https://github.com/Qiskit-Partners/mthree>`_.

    Parameters:
        circuit: Input quantum circuit.

    Returns:
        Mapping of qubits to classical bits for final measurements.
    """
    active_qubits = list(range(circuit.num_qubits))
    active_cbits = list(range(circuit.num_clbits))

    # Find final measurements starting in back
    mapping = {}
    for item in circuit._data[::-1]:
        if item.operation.name == "measure":
            cbit = circuit.find_bit(item.clbits[0]).index
            qbit = circuit.find_bit(item.qubits[0]).index
            if cbit in active_cbits and qbit in active_qubits:
                mapping[qbit] = cbit
                active_cbits.remove(cbit)
                active_qubits.remove(qbit)
        elif item.operation.name != "barrier":
            for qq in item.qubits:
                _temp_qubit = circuit.find_bit(qq).index
                if _temp_qubit in active_qubits:
                    active_qubits.remove(_temp_qubit)

        if not active_cbits or not active_qubits:
            break

    # Sort so that classical bits are in numeric order low->high.
    mapping = dict(sorted(mapping.items(), key=lambda item: item[1]))
    return mapping


T = TypeVar("T")  # pylint: disable=invalid-name


def _finditer(obj: T, objects: list[T]) -> Iterator[int]:
    """Return an iterator yielding the indices matching obj."""
    return map(lambda x: x[0], filter(lambda x: x[1] == obj, enumerate(objects)))


def rng_from_seed(seed: None | int | Generator = None):
    """Build RNG from different seed formats"""
    if seed is None:
        return default_rng()
    if isinstance(seed, Generator):
        return seed
    return default_rng(seed)


def lru_method_cache(*lru_args, **lru_kwargs):
    """Caching decorator for methods in classes"""

    def decorator(func):
        @wraps(func)
        def wrapped_func(self, *args, **kwargs):
            # We're storing the wrapped method inside the instance. If we had
            # a strong reference to self the instance would never die.
            self_weak = ref(self)

            @lru_cache(*lru_args, **lru_kwargs)
            def cached_method(*args, **kwargs):
                return func(self_weak(), *args, **kwargs)

            @wraps(func)
            def wrapped_method(*args, **kwargs):
                func_signature = signature(func)
                bound_args = func_signature.bind(self_weak(), *args, **kwargs)
                bound_args.apply_defaults()
                bound_args = bound_args.arguments
                bound_args.popitem(last=False)
                bound_args = OrderedDict(sorted(bound_args.items()))  # Avoid redundant caching
                return cached_method(**bound_args)

            setattr(self, func.__name__, wrapped_method)
            return wrapped_method(*args, **kwargs)

        return wrapped_func

    return decorator
