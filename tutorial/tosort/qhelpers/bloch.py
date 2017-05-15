"""
Module for generating "bloch"-type experiments from an annotated QASM source.

Author: Andrew Cross
"""
import copy
from qiskit.qasm import Qasm
import qiskit.unroll as unroll
from .blocherror import BlochError


def ez(data, j):
    """Compute the expectation value of Z for the jth qubit.

    The data takes the form {"bitstring":probability,...}
    """
    sum = 0.0
    for k, v in data.items():
        sum += (-1)**int(k[len(k) - j - 1]) * v
    return sum


def make_unrolled_circuit(qasm, basis):
    """Construct a CircuitGraph object for a QASM circuit.

    qasm - string containing QASM source
    basis - list of gate name strings
    """
    ast = Qasm(data=qasm).parse()
    u = unroll.Unroller(ast, unroll.CircuitBackend(basis))
    u.execute()
    return u.be.C


# Register names to use for state tomography subcircuits
bloch_qreg = "q_rep"
bloch_creg = "c"


# State tomography subcircuit for X-basis measurement
blochxcirc = """
OPENQASM 2.0;
gate u2(phi,lambda) q { U(pi/2,phi,lambda) q; }
qreg %(bqreg)s[1];
creg %(bcreg)s[5];
u2(0,pi) %(bqreg)s[0];
measure %(bqreg)s[0] -> %(bcreg)s[0];
""" % {'bqreg': bloch_qreg, 'bcreg': bloch_creg}


# State tomography subcircuit for Y-basis measurement
blochycirc = """
OPENQASM 2.0;
gate u2(phi,lambda) q { U(pi/2,phi,lambda) q; }
gate u1(lambda) q { U(0,0,lambda) q; }
qreg %(bqreg)s[1];
creg %(bcreg)s[5];
u1(-pi/2) %(bqreg)s[0];
u2(0,pi) %(bqreg)s[0];
measure %(bqreg)s[0] -> %(bcreg)s[0];
""" % {'bqreg': bloch_qreg, 'bcreg': bloch_creg}


# State tomography subcircuit for Z-basis measurement
blochzcirc = """
OPENQASM 2.0;
qreg %(bqreg)s[1];
creg %(bcreg)s[5];
measure %(bqreg)s[0] -> %(bcreg)s[0];
""" % {'bqreg': bloch_qreg, 'bcreg': bloch_creg}


def make_bloch_circuits(qasm, basis=["cx", "u1", "u2", "u3", "bloch"]):
    """Generate a collection of circuits for single qubit tomography.

    Generate 3 circuits for each occurrence of a "bloch" opaque gate
    in the input QASM source file.
    qasm - string containing QASM source
    basis - optional list of basic gates

    Returns a list of n experiments for each of the n opaque bloch gates.
    Each experiment is a dict with 3 elements "x", "y", "z" corresponding
    to x, y, and z basis measurement QASM sources and "key" corresponding
    to the parameter tuple for the bloch gate.
    """
    explist = []
    c1 = make_unrolled_circuit(qasm, copy.copy(basis))
    cx = make_unrolled_circuit(blochxcirc, copy.copy(basis))
    cy = make_unrolled_circuit(blochycirc, copy.copy(basis))
    cz = make_unrolled_circuit(blochzcirc, copy.copy(basis))
    nlist = c1.get_named_nodes('bloch')
    for i in range(len(nlist)):
        n = nlist[i]
        nd = c1.G.node[n]
        if nd["type"] != "op" or len(nd["qargs"]) != 1 or \
           len(nd["cargs"]) != 0:
            raise BlochError("expected bloch operation to act on 1 qubit")
        print("bloch #%d -- qubit %s[%d], parameters %s" % (i,
                                                            nd["qargs"][0][0],
                                                            nd["qargs"][0][1],
                                                            nd["params"]))
        blochexp = {"key": nd["params"]}
        for csub in [("x", cx), ("y", cy), ("z", cz)]:
            c1p = c1.deepcopy()
            names = copy.copy(c1p.cregs)
            # Mangle the original creg names so we can ignore them
            for regname in names:
                c1p.rename_register(regname, "aaa" + regname)
            # Make the substitution
            c1p.remove_descendants_of(n)
            c1p.substitute_circuit_one(n, csub[1], [(bloch_qreg, 0)])
            # Clean up left over bloch instructions
            c1p.remove_all_ops_named("bloch")
            c1p.basis.pop("bloch", None)
            c1p.gates.pop("bloch", None)
            blochexp[csub[0]] = c1p.qasm(True)  # qasm with qelib1.inc
        explist.append(blochexp)
    return explist
