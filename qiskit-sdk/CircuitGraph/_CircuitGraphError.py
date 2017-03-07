"""
Exception for errors raised by the CircuitGraph object.

Author: Andrew Cross
"""


class CircuitGraphError(Exception):
    """Base class for errors raised by the circuit graph object."""

    def __init__(self, *msg):
        """Set the error message."""
        self.msg = ' '.join(msg)

    def __str__(self):
        """Return the message."""
        return repr(self.msg)
