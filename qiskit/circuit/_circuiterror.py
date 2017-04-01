"""
Exception for errors raised by the Circuit object.

Author: Andrew Cross
"""


class CircuitError(Exception):
    """Base class for errors raised by the circuit object."""

    def __init__(self, *msg):
        """Set the error message."""
        self.msg = ' '.join(msg)

    def __str__(self):
        """Return the message."""
        return repr(self.msg)
