# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Initialize qubit registers to desired arbitrary state.
"""

import math
import numpy as np
import scipy

from qiskit.exceptions import QiskitError
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.circuit import Gate
from qiskit.dagcircuit import DAGCircuit
from qiskit.extensions.standard.cx import CnotGate
from qiskit.extensions.standard.ry import RYGate
from qiskit.extensions.standard.rz import RZGate

_EPS = 1e-10  # global variable used to chop very small numbers to zero


class InitializeGate(Gate):  # pylint: disable=abstract-method
    """Complex amplitude initialization.

    Class that implements the (complex amplitude) initialization of some
    flexible collection of qubit registers (assuming the qubits are in the
    zero state).
    """
    def __init__(self, params, circ=None):
        """Create new initialize composite gate.
        
        params (list): vector of complex amplitudes to initialize to
        circ (QuantumCircuit): where the initialize instruction is attached
        """
        num_qubits = math.log2(len(params))

        # Check if param is a power of 2
        if num_qubits == 0 or not num_qubits.is_integer():
            raise QiskitError("Desired vector not a positive power of 2.")

        # Check if probabilities (amplitudes squared) sum to 1
        if not math.isclose(sum(np.absolute(params) ** 2), 1.0,
                            abs_tol=_EPS):
            raise QiskitError("Sum of amplitudes-squared does not equal one.")

        num_qubits = int(num_qubits)

        super().__init__("init", num_qubits, params, circ)

    def _define_decompositions(self):
        """Calculate a subcircuit that implements this initialization

        Implements a recursive initialization algorithm, including optimizations,
        from "Synthesis of Quantum Logic Circuits" Shende, Bullock, Markov
        https://arxiv.org/abs/quant-ph/0406176v5

        Additionally implements some extra optimizations: remove zero rotations and
        double cnots.
        """
        self.decomposition = DAGCircuit()
        q = QuantumRegister(self.num_qubits, "q")
        self.decomposition.add_qreg(q)

        # call to generate the circuit that takes the desired vector to zero
        decomposition = self.gates_to_uncompute()

        # remove zero rotations and double cnots
        self.optimize_gates(decomposition)
        
        # invert the circuit to create the desired vector from zero (assuming
        # the qubits are in the zero state)
        self.inverse()

        self._decompositions = [decomposition]

    def nth_qubit_from_least_sig_qubit(self, nth):
        """
        Return the qubit that is nth away from the least significant qubit
        (LSB), so n=0 corresponds to the LSB.
        """
        # if LSB is first (as is the case with the IBM QE) and significance is
        # in order:
        return self.qargs[nth]
        # if MSB is first: return self.qargs[self.num_qubits - 1 - n]
        #  equivalent to self.qargs[-(n+1)]
        # to generalize any mapping could be placed here or even taken from
        # the user

    def gates_to_uncompute(self):
        """
        Call to create a circuit with gates that takes the
        desired vector to zero.
        """
        # kick start the peeling loop, and disentangle one-by-one from LSB to MSB
        remaining_param = self.params

        for i in range(self.num_qubits):
            # work out which rotations must be done to disentangle the LSB
            # qubit (we peel away one qubit at a time)
            (remaining_param,
             thetas,
             phis) = InitializeGate._rotations_to_disentangle(remaining_param)

            # perform the required rotations to decouple the LSB qubit (so that
            # it can be "factored" out, leaving a shorter amplitude vector to peel away)
            self.decomposition.compose_back(self.multiplex(RZGate, i, phis))
            self.decomposition.compose_back(self.multiplex(RYGate, i, thetas))

    @staticmethod
    def _rotations_to_disentangle(local_param):
        """
        Static internal method to work out Ry and Rz rotation angles used
        to disentangle the LSB qubit.
        These rotations make up the block diagonal matrix U (i.e. multiplexor)
        that disentangles the LSB.

        [[Ry(theta_1).Rz(phi_1)  0   .   .   0],
         [0         Ry(theta_2).Rz(phi_2) .  0],
                                    .
                                        .
          0         0           Ry(theta_2^n).Rz(phi_2^n)]]
        """
        remaining_vector = []
        thetas = []
        phis = []

        param_len = len(local_param)

        for i in range(param_len // 2):
            # Ry and Rz rotations to move bloch vector from 0 to "imaginary"
            # qubit
            # (imagine a qubit state signified by the amplitudes at index 2*i
            # and 2*(i+1), corresponding to the select qubits of the
            # multiplexor being in state |i>)
            (remains,
             add_theta,
             add_phi) = InitializeGate._bloch_angles(
                 local_param[2*i: 2*(i + 1)])

            remaining_vector.append(remains)

            # rotations for all imaginary qubits of the full vector
            # to move from where it is to zero, hence the negative sign
            thetas.append(-add_theta)
            phis.append(-add_phi)

        return remaining_vector, thetas, phis

    @staticmethod
    def _bloch_angles(pair_of_complex):
        """
        Static internal method to work out rotation to create the passed in
        qubit from the zero vector.
        """
        [a_complex, b_complex] = pair_of_complex
        # Force a and b to be complex, as otherwise numpy.angle might fail.
        a_complex = complex(a_complex)
        b_complex = complex(b_complex)
        mag_a = np.absolute(a_complex)
        final_r = float(np.sqrt(mag_a ** 2 + np.absolute(b_complex) ** 2))
        if final_r < _EPS:
            theta = 0
            phi = 0
            final_r = 0
            final_t = 0
        else:
            theta = float(2 * np.arccos(mag_a / final_r))
            a_arg = np.angle(a_complex)
            b_arg = np.angle(b_complex)
            final_t = a_arg + b_arg
            phi = b_arg - a_arg

        return final_r * np.exp(1.J * final_t/2), theta, phi

    def multiplex(self, target_gate, target_qubit_index, list_of_angles):
        """
        Internal recursive method to create gates to perform rotations on the
        imaginary qubits: works by rotating LSB (and hence ALL imaginary
        qubits) by combo angle and then flipping sign (by flipping the bit,
        hence moving the complex amplitudes) of half the imaginary qubits
        (CNOT) followed by another combo angle on LSB, therefore executing
        conditional (on MSB) rotations, thereby disentangling bottom qubit (LSB).

        Args:
            target_gate (Gate): Ry or Rz gate to apply to target qubit, multiplexed
                over all other "select" qubits
            target_qubit_index (int): the multiplexor's "data" qubit
            list_of_angles (list[float]): list of rotation angles to apply Ry and Rz

        Returns:
            DAGCircuit: the circuit implementing the multiplexor's action
        """
        list_len = len(list_of_angles)
        local_num_qubits = int(math.log2(list_len)) + 1
        control_qubit_index = local_num_qubits - 1 + target_qubit_index

        # build multiplex circuit
        # TODO: simplify code with better circuit building API
        multiplex_circuit = DAGCircuit()
        multiplex_circuit.name = "multiplex" + local_num_qubits.__str__()
        q = QuantumRegister()
        multiplex_circuit.add_qreg(self.decomposition.qregs.values[0])

        # Case of no multiplexing = base case for recursion
        if list_len == 1:
            multiplex_circuit.apply_operation_back(
                    target_gate(list_of_angles[0]), [target_qubit], [])
            return multiplex_circuit

        # calc angle weights, assuming recursion (that is the lower-level
        # requested angles have been correctly implemented by recursion
        angle_weight = scipy.kron([[0.5, 0.5], [0.5, -0.5]],
                                  np.identity(2 ** (local_num_qubits - 2)))

        # calc the combo angles
        list_of_angles = angle_weight.dot(np.array(list_of_angles)).tolist()Z

        # recursive step on half the angles fulfilling the above assumption
        multiplex_circuit.compose_back(
            self.multiplex(target_gate, target_qubit,
                            list_of_angles[0:(list_len // 2)]))

        # attach CNOT as follows, thereby flipping the LSB qubit
        multiplex_circuit.apply_operation_back(
                CnotGate(), [control_qubit, target_qubit], [])

        # implement extra efficiency from the paper of cancelling adjacent
        # CNOTs (by leaving out last CNOT and reversing (not inverting) the
        # second lower-level multiplex)
        sub_circuit = self.multiplex(
            target_gate, target_qubit, list_of_angles[(list_len // 2):])
        if list_len > 1:
            multiplex_circuit.compose_back(sub_circuit.reverse())
        else:
            multiplex_circuit.compose_back(sub_circuit)

        # outer multiplex keeps final CNOT, because no adjacent CNOT to cancel
        # with
        if self.num_qubits == local_num_qubits + target_qubit_index:
            multiplex_circuit.apply_operation_back(
                    CnotGate(), [control_qubit, target_qubit], [])

        return multiplex_circuit

    @staticmethod
    def chop_num(numb):
        """
        Set very small numbers (as defined by global variable _EPS) to zero.
        """
        return 0 if abs(numb) < _EPS else numb


# ###############################################################
# Add needed functionality to other classes (it feels
# weird following the Qiskit convention of adding functionality to other
# classes like this ;),
#  TODO: multiple inheritance might be better?)


def reverse(self):
    """
    Reverse (recursively) the sub-gates of this CompositeGate. Note this does
    not invert the gates!
    """
    new_data = []
    for gate in reversed(self.data):
        if isinstance(gate, CompositeGate):
            new_data.append(gate.reverse())
        else:
            new_data.append(gate)
    self.data = new_data

    # not just a high-level reverse:
    # self.data = [gate for gate in reversed(self.data)]

    return self


QuantumCircuit.reverse = reverse


def optimize_gates(self):
    """Remove Zero rotations and Double CNOTS."""
    self.remove_zero_rotations()
    while self.remove_double_cnots_once():
        pass


QuantumCircuit.optimize_gates = optimize_gates


def remove_zero_rotations(self):
    """
    Remove Zero Rotations by looking (recursively) at rotation gates at the
    leaf ends.
    """
    # Removed at least one zero rotation.
    zero_rotation_removed = False
    new_data = []
    for gate in self.data:
        if isinstance(gate, CompositeGate):
            zero_rotation_removed |= gate.remove_zero_rotations()
            if gate.data:
                new_data.append(gate)
        else:
            if ((not isinstance(gate, Gate)) or
                    (not (gate.name == "rz" or gate.name == "ry" or
                          gate.name == "rx") or
                     (InitializeGate.chop_num(gate.params[0]) != 0))):
                new_data.append(gate)
            else:
                zero_rotation_removed = True

    self.data = new_data

    return zero_rotation_removed


QuantumCircuit.remove_zero_rotations = remove_zero_rotations


def number_atomic_gates(self):
    """Count the number of leaf gates. """
    num = 0
    for gate in self.data:
        if isinstance(gate, CompositeGate):
            num += gate.number_atomic_gates()
        else:
            if isinstance(gate, Gate):
                num += 1
    return num


QuantumCircuit.number_atomic_gates = number_atomic_gates


def remove_double_cnots_once(self):
    """
    Remove Double CNOTS paying attention that gates may be neighbours across
    Composite Gate boundaries.
    """
    num_high_level_gates = len(self.data)

    if num_high_level_gates == 0:
        return False
    else:
        if num_high_level_gates == 1 and isinstance(self.data[0],
                                                    CompositeGate):
            return self.data[0].remove_double_cnots_once()

    # Removed at least one double cnot.
    double_cnot_removed = False

    # last gate might be composite
    if isinstance(self.data[num_high_level_gates - 1], CompositeGate):
        double_cnot_removed = \
            double_cnot_removed or\
            self.data[num_high_level_gates - 1].remove_double_cnots_once()

    # don't start with last gate, using reversed so that can del on the go
    for i in reversed(range(num_high_level_gates - 1)):
        if isinstance(self.data[i], CompositeGate):
            double_cnot_removed =\
                double_cnot_removed \
                or self.data[i].remove_double_cnots_once()
            left_gate_host = self.data[i].last_atomic_gate_host()
            left_gate_index = -1
            # TODO: consider adding if semantics needed:
            # to remove empty composite gates
            # if left_gate_host == None: del self.data[i]
        else:
            left_gate_host = self.data
            left_gate_index = i

        if ((left_gate_host is not None) and
                left_gate_host[left_gate_index].name == "cx"):
            if isinstance(self.data[i + 1], CompositeGate):
                right_gate_host = self.data[i + 1].first_atomic_gate_host()
                right_gate_index = 0
            else:
                right_gate_host = self.data
                right_gate_index = i + 1

            if (right_gate_host is not None) \
                    and right_gate_host[right_gate_index].name == "cx" \
                    and (left_gate_host[left_gate_index].qargs ==
                         right_gate_host[right_gate_index].qargs):
                del right_gate_host[right_gate_index]
                del left_gate_host[left_gate_index]
                double_cnot_removed = True

    return double_cnot_removed


QuantumCircuit.remove_double_cnots_once = remove_double_cnots_once


def first_atomic_gate_host(self):
    """Return the host list of the leaf gate on the left edge."""
    if self.data:
        if isinstance(self.data[0], CompositeGate):
            return self.data[0].first_atomic_gate_host()
        return self.data

    return None


QuantumCircuit.first_atomic_gate_host = first_atomic_gate_host


def last_atomic_gate_host(self):
    """Return the host list of the leaf gate on the right edge."""
    if self.data:
        if isinstance(self.data[-1], CompositeGate):
            return self.data[-1].last_atomic_gate_host()
        return self.data

    return None


QuantumCircuit.last_atomic_gate_host = last_atomic_gate_host


def initialize(self, params, qubits):
    """Apply initialize to circuit."""
    # TODO: make initialize an Instruction, and insert reset
    # TODO: avoid explicit reset if compiler determines a |0> state
    return self._attach(InitializeGate(params, self), qubits, [])


QuantumCircuit.initialize = initialize
