from sympy.physics.quantum.gate import u, H, X, Y, Z, S, T, CNOT, IdentityGate, OneQubitGate, TwoQubitGate, Gate, XGate, CGate, UGate
from sympy.physics.quantum.represent import represent
# from sympy.physics.quantum import Dagger

from sympy import pprint, pretty, symbols, Matrix, pi, E, I, cos, sin, N, exp

# this file is a centralized control of the gate domain
# any extension to the domain should be made here

# needed in qchecker.py. tell me the basis gates that should be recognized, otherwise, they will be translated to U gates
_basic_gates_string_IBM="id,x,y,z,h,s,sdg,cx,t,tdg" #maybe block u/cu gates?
_basic_gates_string_IBM_advanced="id,x,y,z,h,s,sdg,cx,t,tdg,u1,u2,u3,cy,cz,ccx,cu1,cu3"



error_margin = 0.01
def regulate(theta):
    if abs(N(theta - pi)) < error_margin:
        return pi, True
    elif abs(N(theta - pi/2)) < error_margin:
        return pi/2, True
    elif abs(N(theta - pi/4)) < error_margin:
        return pi/4, True
    elif abs(N(theta- 2*pi)) < error_margin:
        return 2*pi, True
    else:
        return theta, theta == 0 # if theta ==0, we also think it is regular



# extensible gate domain:
class SDGGate(OneQubitGate):
    gate_name = u('SDG')
    def get_target_matrix(self, format='sympy'):
        return Matrix([[1, 0], [0, -I]])
    # def _eval_commutator_ZGate(self, other, **hints):
    #     return Integer(0)
    # def _eval_commutator_TGate(self, other, **hints):
    #     return Integer(0)


class TDGGate(OneQubitGate):
    gate_name = u('TDG')
    def get_target_matrix(self, format='sympy'):
        return Matrix([[1, 0], [0, exp(-I*pi/4)]])
    # def _eval_commutator_ZGate(self, other, **hints):
    #     return Integer(0)
    # def _eval_commutator_PhaseGate(self, other, **hints):
    #     return Integer(0)



#ugate is explained in page 5, eq (2) of https://github.com/IBM/qiskit-openqasm/blob/master/spec/qasm2.pdf
# besides, page 11 says that even x/y/z are internally implemented with ugate
# note that z != U(0,0,pi) despite z is implemented using U(0,0,pi)
# mathematicians would complain because the final results using the two gates would not be the same
# physicists say it is fine since they differ only in the "global phase", i.e., the e^iw constant in front of U
# global phase is useless in quantum because it is subject to the origin time you defined.
# for this reason, the global phase (e^iw) is completely ignored in the U gate
# as a consequence, when you check whether two states are equivalent, you should not check the mathematical equality.
# instead, you should check if one state can be converted to the other by multiplying certain e^iw.
def compute_ugate_matrix(parafloatlist):
    theta = parafloatlist[0]
    phi = parafloatlist[1]
    lamb = parafloatlist[2]

    theta, theta_is_regular = regulate(theta)
    phi, phi_is_regular = regulate(phi)
    lamb, lamb_is_regular = regulate(lamb)


    uMat = Matrix([[(E**(-I*(phi+lamb)/2)) * cos(theta/2), (-E**(-I*(phi-lamb)/2)) * sin(theta/2)],
                   [(E**(I*(phi-lamb)/2)) * sin(theta/2), (E**(I*(phi+lamb)/2)) * cos(theta/2)]])

    if theta_is_regular and phi_is_regular and lamb_is_regular: # regular: we do not need concrete float value
        uMatNumeric = uMat
    else:
        uMatNumeric = uMat.evalf()
    return uMatNumeric


def get_sym_op(name, qid_tuple):
    if name == 'ID':
        return IdentityGate(*qid_tuple) # de-tuple means unpacking
    elif name == 'X':
        return X(*qid_tuple)
    elif name == 'Y':
        return Y(*qid_tuple)
    elif name == 'Z':
        return Z(*qid_tuple)
    elif name == 'H':
        return H(*qid_tuple)
    elif name == 'S':
        return S(*qid_tuple)
    elif name == 'SDG':
        return SDGGate(*qid_tuple)
    elif name == 'T':
        return T(*qid_tuple)
    elif name == 'TDG':
        return TDGGate(*qid_tuple)
    elif name == 'CX' or name == 'CNOT':
        return CNOT(*qid_tuple)
    elif name == 'CY':
        return CGate(qid_tuple[0], Y(qid_tuple[1])) # qid_tuple: control target
    elif name == 'CZ':
        return CGate(qid_tuple[0], Z(qid_tuple[1])) # qid_tuple: control target
    elif name == 'CCX' or name == 'CCNOT' or name == 'TOFFOLI':
        return CGate((qid_tuple[0], qid_tuple[1]), X(qid_tuple[2])) # qid_tuple: control1, control2, target
    else: # U gate or CU gate
        if name.startswith('U') or name.startswith('CU'):
            inside = name[name.find('(')+1:name.find(')')]
            paralist = inside.split(',')
            parafloatlist = []
            for tmp in paralist:
                parafloatlist.append(float(tmp))

            if len(parafloatlist) == 1: # [theta=0, phi=0, lambda]
                parafloatlist.insert(0, 0.0)
                parafloatlist.insert(0, 0.0)
            elif len(parafloatlist) == 2: #[theta=pi/2, phi, lambda]
                parafloatlist.insert(0, pi/2)
            elif len(parafloatlist) == 3: #[theta, phi, lambda]
                pass
            else:
                return NotImplemented

            uMat = compute_ugate_matrix(parafloatlist)
            class UGatePeng(OneQubitGate):
                    gate_name = u('U')
                    def get_target_matrix(self, format='sympy'):
                        return uMat
            # the original UGate in sympy does not accept the matrix with numerical values
            if name.startswith('U'):
                return UGatePeng(*qid_tuple) # the first arg of UGate should be a tuple of qubits to be applied to
            elif name.startswith('CU'): # additional treatment for CU1, CU2, CU3
                return CGate(qid_tuple[0], UGatePeng(*qid_tuple[1:]))
        elif name == "MEASURE":
            return None # do nothing...
        else:
            raise Exception('Not supported')



# some additional use of gate names appear in plot_quantum_circuit.py
