=======
QkParam
=======

Represents a circuit parameter which may hold real or symbolic values.

When added to a ``QkCircuit``, the values of all parameters must be determined,
that is, all values must be real by the time the circuit starts execution.

While functionality for a ``QkParam`` is currently limited, a user is allowed
to do several essential operations similarly to the ones allowed by :class:`.Parameter`.

Functions
=========

.. doxygengroup:: QkParam
    :members:
    :content-only:
