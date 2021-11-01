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

"""An iterator that returns the first item and an iterator over the remaining items of an iterable."""

def peel(iterable):
    """Return a tuple containing the first item in iterable and an iterator
    over the remaining items.
    """
    iterator = iter(iterable)
    first = next(iterator)
    return  first, iterator
