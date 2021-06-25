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

# pylint: disable=not-callable

"""
Base register reference object.
"""
import re
import itertools
import numpy as np

from qiskit.circuit.exceptions import CircuitError


class Register:
    """Implement a generic register."""

    __slots__ = ["_name", "_size", "_bits", "_hash", "_repr"]

    # Register name should conform to OpenQASM 2.0 specification
    # See appendix A of https://arxiv.org/pdf/1707.03429v2.pdf
    name_format = re.compile("[a-z][a-zA-Z0-9_]*")

    # Counter for the number of instances in this class.
    instances_counter = itertools.count()
    # Prefix to use for auto naming.
    prefix = "reg"
    bit_type = None

    def __init__(self, size=None, name=None, bits=None):
        """Create a new generic register.

        Either the ``size`` or the ``bits`` argument must be provided. If
        ``size`` is not None, the register will be pre-populated with bits of the
        correct type.

        Args:
            size (int): Optional. The number of bits to include in the register.
            name (str): Optional. The name of the register. If not provided, a
               unique name will be auto-generated from the register type.
            bits (list[Bit]): Optional. A list of Bit() instances to be used to
               populate the register.

        Raises:
            CircuitError: if both the ``size`` and ``bits`` arguments are
                provided, or if neither are.
            CircuitError: if ``size`` is not valid.
            CircuitError: if ``name`` is not a valid name according to the
                OpenQASM spec.
            CircuitError: if ``bits`` contained bits of an incorrect type.
        """

        if (size, bits) == (None, None) or (size is not None and bits is not None):
            raise CircuitError(
                "Exactly one of the size or bits arguments can be "
                "provided. Provided size=%s bits=%s." % (size, bits)
            )

        # validate (or cast) size
        if bits is not None:
            size = len(bits)

        try:
            valid_size = size == int(size)
        except (ValueError, TypeError):
            valid_size = False

        if not valid_size:
            raise CircuitError(
                "Register size must be an integer. (%s '%s' was provided)"
                % (type(size).__name__, size)
            )
        size = int(size)  # cast to int

        if size <= 0:
            raise CircuitError(
                "Register size must be positive (%s '%s' was provided)"
                % (type(size).__name__, size)
            )

        # validate (or cast) name
        if name is None:
            name = "%s%i" % (self.prefix, next(self.instances_counter))
        else:
            try:
                name = str(name)
            except Exception as ex:
                raise CircuitError(
                    "The circuit name should be castable to a string "
                    "(or None for autogenerate a name)."
                ) from ex
            if self.name_format.match(name) is None:
                raise CircuitError(
                    "%s is an invalid OPENQASM register name. See appendix"
                    " A of https://arxiv.org/pdf/1707.03429v2.pdf." % name
                )

        self._name = name
        self._size = size

        self._hash = hash((type(self), self._name, self._size))
        self._repr = "%s(%d, '%s')" % (self.__class__.__qualname__, self.size, self.name)
        if bits is not None:
            # pylint: disable=isinstance-second-argument-not-valid-type
            if any(not isinstance(bit, self.bit_type) for bit in bits):
                raise CircuitError(
                    "Provided bits did not all match " "register type. bits=%s" % bits
                )
            self._bits = list(bits)
        else:
            self._bits = [self.bit_type(self, idx) for idx in range(size)]

    @property
    def name(self):
        """Get the register name."""
        return self._name

    @property
    def size(self):
        """Get the register size."""
        return self._size

    def __repr__(self):
        """Return the official string representing the register."""
        return self._repr

    def __len__(self):
        """Return register size."""
        return self._size

    def __getitem__(self, key):
        """
        Arg:
            bit_type (Qubit or Clbit): a constructor type return element/s.
            key (int or slice or list): index of the clbit to be retrieved.

        Returns:
            Qubit or Clbit or list(Qubit) or list(Clbit): a Qubit or Clbit instance if
            key is int. If key is a slice, returns a list of these instances.

        Raises:
            CircuitError: if the `key` is not an integer or not in the range `(0, self.size)`.
        """
        if not isinstance(key, (int, np.integer, slice, list)):
            raise CircuitError("expected integer or slice index into register")
        if isinstance(key, slice):
            return self._bits[key]
        elif isinstance(key, list):  # list of qubit indices
            if max(key) < len(self):
                return [self._bits[idx] for idx in key]
            else:
                raise CircuitError("register index out of range")
        else:
            return self._bits[key]

    def __iter__(self):
        for idx in range(self._size):
            yield self._bits[idx]

    def __eq__(self, other):
        """Two Registers are the same if they are of the same type
        (i.e. quantum/classical), and have the same name and size. Additionally,
        if either Register contains new-style bits, the bits in both registers
        will be checked for pairwise equality. If two registers are equal,
        they will have behave identically when specified as circuit args.

        Args:
            other (Register): other Register

        Returns:
            bool: `self` and `other` are equal.
        """
        if self is other:
            return True

        res = False
        if (
            type(self) is type(other)
            and self._repr == other._repr
            and all(
                # For new-style bits, check bitwise equality.
                sbit == obit
                for sbit, obit in zip(self, other)
                if None in (sbit._register, sbit._index, obit._register, obit._index)
            )
        ):
            res = True
        return res

    def __hash__(self):
        """Make object hashable, based on the name and size to hash."""
        return self._hash
