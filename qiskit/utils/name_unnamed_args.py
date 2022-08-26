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

"""Tool to name unnamed arguments."""

import functools


def name_args(mapping, skip=0):
    """Decorator to convert unnamed arguments to named ones.

    Can be used to deprecate old signatures of a function, e.g.

    .. code-block::

        old_f(a: TypeA, b: TypeB, c: TypeC)
        new_f(a: TypeA, d: TypeD, b: TypeB=None, c: TypeC=None)

    Then, to support the old signature this decorator can be used as

    .. code-block::

        @name_args([
            ('a'),  # stays the same
            ('d', {TypeB: 'b'}),  # if arg is of type TypeB, call if 'b' else 'd'
            ('b', {TypeC: 'c'})
        ])
        def new_f(a: TypeA, d: TypeD, b: TypeB=None, c: TypeC=None):
            if b is not None:
                # raise warning, this is deprecated!
            if c is not None:
                # raise warning, this is deprecated!

    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # turn args into kwargs
            for arg, replacement in zip(args[skip:], mapping):
                default_name = replacement[0]
                if len(replacement) == 1:  # just renaming, no special cases
                    if default_name in kwargs:
                        raise ValueError(f"Name collapse on {default_name}")
                    kwargs[default_name] = arg
                else:
                    # check if we find a special name
                    name = None
                    for special_type, special_name in replacement[1].items():
                        if isinstance(arg, special_type):
                            name = special_name
                            break
                    if name is None:
                        name = default_name

                    if name in kwargs:
                        raise ValueError(f"Name collapse on {default_name}")
                    kwargs[name] = arg

            return func(*args[:skip], **kwargs)

        return wrapper

    return decorator
