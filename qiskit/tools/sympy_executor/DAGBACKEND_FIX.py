from qiskit import unroll

class DAGBACKEND_FIX(unroll.DAGBackend):
    def start_gate(self, name, args, qubits, nested_scope=None):
        """Begin a custom gate.

        name is name string.
        args is list of Node expression objects.
        qubits is list of (regname, idx) tuples.
        nested_scope is a list of dictionaries mapping expression variables
        to Node expression objects in order of increasing nesting depth.
        """
        if self.listen and name not in self.basis \
           and self.gates[name]["opaque"]:
            raise unroll.BackendError("opaque gate %s not in basis" % name)
        if self.listen and name in self.basis:
            if self.creg is not None:
                condition = (self.creg, self.cval)
            else:
                condition = None
            self.in_gate = name
            self.listen = False
            self.circuit.add_basis_element(name, len(qubits), 0, len(args))
            self.circuit.apply_operation_back(
                name, qubits, [], list(map(lambda x: x.real(nested_scope),
                                           args)), condition)
