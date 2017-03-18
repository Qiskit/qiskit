"""
Base register object.

Author: Andrew Cross
"""
from ._qiskitexception import QISKitException


class Register(object):
    """Implement a generic register."""

    def __init__(self, name, sz):
        """Create a new generic register."""
        self.name = name
        self.sz = sz
        self.bound_to = []
        self.data = []
        if sz <= 0:
            raise QISKitException("register size must be positive")

    def bind_to(self, prog):
        """Bind register to program."""
        # TODO: We are binding self to prog. self already has data.
        # one of the gates g involves register y. y is not bound to prog.
        # y also has some data. for example y's data may have gates
        # that preceed g. we clearly need to add y. we need to add the
        # preceeding gates. those gates may involve further registers ...
        # the complication to recursion is that y's data and self's data
        # are not disjoint if there is another gate g' in self's data
        # that involves y, since y's data will contain this too. ??????
        if prog not in self.bound_to:
            self.bound_to.append(prog)

    def _check_range(self, j):
        """Check that j is a valid index into self."""
        if j < 0 or j >= self.sz:
            raise QISKitException("register index out of range")
