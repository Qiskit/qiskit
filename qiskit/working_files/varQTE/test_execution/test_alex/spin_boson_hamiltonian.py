from qiskit import QuantumCircuit, QuantumRegister
from qiskit.opflow import OperatorBase, CircuitOp

import numpy as np
from typing import Optional, Union, List
import warnings
import copy



class SpinBosonHamiltonian:
    """Defining spin-boson Hamiltonian in various representations.
    1) Matrix representation (NOTE: currently onle 1 fermionic + n bosonic
                                    modes with 2 modals each are supported)
    2) quantum circuit representation entangled with ancilla qubit as needed
       for variational real time evolution.

    A direct boson-to-qubit mapping (SES) is used (e.g., 1 fermion, 2 bosons,
    2 modals per bosonic mode, equals 1 + 2*2 = 5 qubits)

    For more information on spin-boson models in quantum circuits, see
    http://arxiv.org/abs/1909.08640
    For a theoretical investigation, see, e.g.,
    https://arxiv.org/abs/1512.04244
    https://arxiv.org/abs/1701.04709
    """

    def __init__(self,
                 num_particles: Union[List[int], int],
                 num_modals: int,
                 omega: Optional[Union[List[float], float]],
                 g_couple: Optional[Union[List[float], float]],
                 delta: float,
                 epsilon: float,
                 coupling: str = 'x'):
        if isinstance(num_particles, list):
            if num_particles[0] != 1:
                warnings.warn('Currently only 1 Fermion is supported! '
                              'Setting fermion number to 1')
            self._num_fermions = 1
            self._num_bosons = num_particles[1]
        else:
            self._num_fermions = 1
            self._num_bosons = num_particles - 1
        self._num_modals = num_modals

        self._num_qubits = (self._num_fermions
                            + self._num_bosons*self._num_modals)

        self._epsilon = epsilon
        self._delta = delta
        if (coupling == 'x') or (coupling == 'y') or (coupling == 'z'):
            self._coupling = coupling
        else:
            raise ValueError('If supplied as an argument, coupling must be '
                             'of type string and either be "x", "y", or "z"')

        if isinstance(omega, list) and len(omega) != self._num_bosons:
            raise ValueError('If omega is supplied as list, its length must '
                             'match number of bosons')
        elif isinstance(omega, list) and len(omega) == self._num_bosons:
            self._omega = omega
        else:
            self._omega = [omega]*self._num_bosons
            print('Taking the same self energy for all modes in H: '
                  'omega = {}'.format(self._omega))

        if isinstance(g_couple, list) and len(g_couple) != self._num_bosons:
            raise ValueError('If g_couple is supplied as list, its length must '
                             'match number of bosons')
        elif isinstance(g_couple, list) and len(g_couple) == self._num_bosons:
            self._g_couple = g_couple
        else:
            self._g_couple = [g_couple]*self._num_bosons
            print('Taking the same coupling strength for all modes in H: '
                  'g_couple = {}'.format(self._g_couple))


    @property
    def num_particles(self):
        return [self._num_fermions, self._num_bosons]

    @property
    def num_modals(self):
        return self._num_modals

    @property
    def num_qubits(self):
        return self._num_qubits

    @property
    def omega(self):
        return self._omega

    @property
    def omega(self):
        return self._omega

    @property
    def delta(self):
        return self._delta

    @property
    def epsilon(self):
        return self._epsilon

    @property
    def g_couple(self):
        return self._g_couple

    @property
    def coupling(self):
        return self._coupling

    def get_all_hamiltonian_params(self):
        return {'omega': self._omega, 'g_couple': self._g_couple,
                'epsilon': self._epsilon, 'delta': self._delta}



    # --------------------------------
    # -*- List of Qiskit-Operators -*-
    # --------------------------------

    def as_circ_list(self, q=None):
        """
        """

        if q is None:
            fr = QuantumRegister(self._num_fermions, 'f')
            br = QuantumRegister(self._num_bosons*self._num_modals, 'b')
        else:
            fr = q[0:self._num_fermions]
            br = q[self._num_fermions:]

        circ_list = []
        # init_circ = QuantumCircuit(fr, br)

        # boson self-terms
        for k in range(self._num_bosons):
            for nk in range(self._num_modals-1):
                fac = 1/4*self._omega[k]*(nk + 1)

                circ = QuantumCircuit(fr, br)
                circ.z(br[k*self._num_modals + nk])
                circ_list.append((fac, copy.deepcopy(circ)))

                circ = QuantumCircuit(fr, br)
                circ.z(br[k*self._num_modals + nk + 1])
                circ_list.append((-fac, copy.deepcopy(circ)))

                circ = QuantumCircuit(fr, br)
                circ.z(br[k*self._num_modals + nk])
                circ.z(br[k*self._num_modals + nk + 1])
                circ_list.append((-fac, copy.deepcopy(circ)))

                # same as 0.5*id(br[nk]) + 0.5*id(br[nk+1])
                circ = QuantumCircuit(fr, br)
                # circ.i(br[k*self._num_modals + nk])
                circ_list.append((fac, copy.deepcopy(circ)))

        # Fermion terms
        circ = QuantumCircuit(fr, br)
        circ.z(fr[0])
        circ_list.append((0.5*self._epsilon, copy.deepcopy(circ)))

        circ = QuantumCircuit(fr, br)
        circ.x(fr[0])
        circ_list.append((self._delta, copy.deepcopy(circ)))

        # Coupling terms
        for k in range(self._num_bosons):
            for nk in range(self._num_modals-1):
                fac = 0.5*self._g_couple[k]*np.sqrt(nk + 1)

                circ = QuantumCircuit(fr, br)
                circ.x(br[k*self._num_modals + nk])
                circ.x(br[k*self._num_modals + nk + 1])
                if self._coupling == 'x':
                    circ.x(fr[0])
                elif self._coupling == 'y':
                    circ.y(fr[0])
                elif self._coupling == 'z':
                    circ.z(fr[0])

                circ_list.append((fac, copy.deepcopy(circ)))

                circ = QuantumCircuit(fr, br)
                circ.y(br[k*self._num_modals + nk])
                circ.y(br[k*self._num_modals + nk + 1])
                if self._coupling == 'x':
                    circ.x(fr[0])
                elif self._coupling == 'y':
                    circ.y(fr[0])
                elif self._coupling == 'z':
                    circ.z(fr[0])

                circ_list.append((fac, copy.deepcopy(circ)))


        return circ_list



    def as_summed_op(self, q=None):
        circ_list = self.as_circ_list(q=q)

        summed_H = 0
        for c in circ_list:
            summed_H += c[0]*CircuitOp(primitive=c[1])

        return summed_H



    def as_matrix(self, q=None):
        H_op = self.as_summed_op(q=q)

        return H_op.to_matrix()



    # ------------------------------------------
    # -*- Trotter-Circuit for Time-Evolution -*-
    # ------------------------------------------

    def _build_controlled_two_qubit_gate(self, ctrl_q, tar_q, param):
        """Constucts controlled two-qubit gates with control qubit in fermion
        register and two target qubits in boson register.

        Args:
            ctrl_q: control qubit in fermion register
            tar_q: first of the two adjecent target qubits in boson register
            param: variational parameter
            param2: variational parameter, different from param if
                    self._adjoint_params_distinct == True

        Returns:
            quantum circuit consisting of controlled two-qubit gate only
        """

        assert ((tar_q+1) % self._num_modals != 0), ('First target of two-qubit'
            'gate cannot be last qubit in k-mode (k-th boson) register')

        fr = QuantumRegister(self._num_fermions, 'f')
        br = QuantumRegister(self._num_bosons*self._num_modals, 'b')
        twoQgate = QuantumCircuit(fr, br)

        twoQgate.h(br[tar_q]) # rotates Z to X basis
        twoQgate.h(br[tar_q + 1]) # rotates Z to Y basis
        twoQgate.cx([fr[ctrl_q], br[tar_q]], [br[tar_q], br[tar_q+1]])
        twoQgate.rz(param, br[tar_q + 1])
        twoQgate.cx([br[tar_q], fr[ctrl_q]], [br[tar_q+1], br[tar_q]])
        twoQgate.h(br[tar_q + 1]) # rotates Y to Z basis
        twoQgate.h(br[tar_q]) # rotates X to Z basis

        twoQgate.rx(-np.pi/2, br[tar_q]) # rotates Z to Y basis
        twoQgate.rx(-np.pi/2, br[tar_q + 1]) # rotates Z to X basis
        twoQgate.cx([fr[ctrl_q], br[tar_q]], [br[tar_q], br[tar_q+1]])
        twoQgate.rz(param, br[tar_q + 1]) # Rz^+
        twoQgate.cx([br[tar_q], fr[ctrl_q]], [br[tar_q+1], br[tar_q]])
        twoQgate.rx(np.pi/2, br[tar_q + 1]) # rotates X to Z basis
        twoQgate.rx(np.pi/2, br[tar_q]) # rotates Y to Z basis

        return twoQgate


    def _construct_all_trotter_circs(self, dt, trotter, order, q):
        """
        """

        if q is None:
            fr = QuantumRegister(self._num_fermions, 'f')
            br = QuantumRegister(self._num_bosons*self._num_modals, 'b')
        else:
            fr = q[0:self._num_fermions]
            br = q[self._num_fermions:]

        circ_list = []

        trotter_depth_global = trotter[0]
        trotter_depth_local = trotter[1]

        if order == 1:
            dt_new = dt / (trotter_depth_global)
        elif order == 2:
            dt_new = dt / (2*trotter_depth_global)

        # NOTE: do for r in range(trotter_depth_global) in method below.
        circ = QuantumCircuit(fr, br)
        if self._coupling == 'x':
            circ.h(fr[0])
            circ_list.append(circ)
        elif self._coupling == 'y':
            # NOTE: if reverse order of Trotter series is needed for Trotter
            # order > 1, need to change signs in these RX-gates!
            warnings.warn('CAREFUL: if reverse order of Trotter series is'
                          'needed for Trotter order > 1, need to change signs'
                          'in these RX-gates!')
            circ.rx(-np.pi/2, fr[0])
            circ_list.append(circ)

        for k in range(self._num_bosons):
            for s in range(trotter_depth_local):
                # append exp(... X_even) to circuit
                circ = QuantumCircuit(fr, br)
                for nk in range(0, self._num_modals - 1, 2):
                    gate_param = (dt_new*self._g_couple[k]*np.sqrt(nk+1)
                                  /(trotter_depth_local))

                    circ.compose(self._build_controlled_two_qubit_gate(
                                    0, k*self._num_modals + nk, gate_param),
                                 inplace=True)
                circ_list.append(circ)

                # append exp(... X_odd) to circuit
                circ = QuantumCircuit(fr, br)
                for nk in range(1, self._num_modals - 1, 2):
                    gate_param = (dt_new*self._g_couple[k]*np.sqrt(nk+1)
                                  /(trotter_depth_local))

                    circ.compose(self._build_controlled_two_qubit_gate(
                                    0, k*self._num_modals + nk, gate_param),
                                inplace=True)
                circ_list.append(circ)

        circ = QuantumCircuit(fr, br)
        if self._coupling == 'x':
            circ.h(fr[0])
            circ_list.append(circ)
        elif self._coupling == 'y':
            # NOTE: if reverse order of Trotter series is needed for Trotter
            # order > 1, need to change signs in these RX-gates!
            circ.rx(np.pi/2, fr[0])
            circ_list.append(circ)

        ### Construct fermion terms of Hamiltonian exponential
        # NOTE: Gate of form exp(-i/2 param sigma)
        # Thus there is a factor 2 for delta and no factor 0.5 for epsilon
        # Delta*sig_x
        circ = QuantumCircuit(fr, br)
        circ.rx(dt_new*2*self._delta, fr[0])
        circ_list.append(circ)

        # epsilon/2 * sig_z
        circ = QuantumCircuit(fr, br)
        circ.rz(dt_new*self._epsilon, fr[0])
        circ_list.append(circ)

        # NOTE: Gate param has factor 1/2 instead of 1/4 due to gate
        # definition exp(-i/2 param sigma), as with delta and epsilon terms
        circ = QuantumCircuit(fr, br)
        for k in range(self._num_bosons):
            for nk in range(self._num_modals - 1):
                gate_param = (dt_new*self._omega[k]*(nk + 1)*0.5)
                bos_q = k*self._num_modals + nk
                circ.cx(br[bos_q], br[bos_q + 1])
                circ.rz(-gate_param, br[bos_q + 1])
                circ.cx(br[bos_q], br[bos_q + 1])
                circ.rz(gate_param, br[bos_q])
                circ.rz(-gate_param, br[bos_q + 1])
        circ_list.append(circ)

        return circ_list


    def init_circ(self, q: QuantumRegister = None):
        """
        """

        if q is None:
            fr = QuantumRegister(self._num_fermions, 'f')
            br = QuantumRegister(self._num_bosons*self._num_modals, 'b')
        elif len(q) != self._num_qubits:
            raise ValueError('If passed as an argument, the number of qubits '
                             'has to match requirement N_q = N_ferm + '
                             'N_bos * N_modals, in this case, '
                             '{}'.format(self._num_qubits))
        else:
            fr = q[0:self._num_fermions]
            br = q[self._num_fermions:]

        circ = QuantumCircuit(fr, br)

        if (q is None):
            for k in range(self._num_bosons):
                circ.x(br[k*self._num_modals]) # initialize k-mode to |01..0>

        return circ


    def construct_trotter_circuit(self,
                                  dt: float,
                                  trotter: List[int],
                                  order: int,
                                  reverse: bool = False,
                                  init_reg_to_zero: bool = False,
                                  q: QuantumRegister = None):
        """
        """

        if q is None:
            fr = QuantumRegister(self._num_fermions, 'f')
            br = QuantumRegister(self._num_bosons*self._num_modals, 'b')
        elif len(q) != self._num_qubits:
            raise ValueError('If passed as an argument, the number of qubits '
                             'has to match requirement N_q = N_ferm + '
                             'N_bos * N_modals, in this case, '
                             '{}'.format(self._num_qubits))
        else:
            fr = q[0:self._num_fermions]
            br = q[self._num_fermions:]

        trotter_depth_global = trotter[0]

        circ = QuantumCircuit(fr, br)

        if (q is None) and (not init_reg_to_zero):
            for k in range(self._num_bosons):
                circ.x(br[k*self._num_modals]) # initialize k-mode to |01..0>

        circ_list = self._construct_all_trotter_circs(dt, trotter, order, q)
        for r in range(trotter_depth_global):
            if not reverse:
                for c in circ_list:
                    circ.compose(c, inplace=True)
            else:
                for c in circ_list[::-1]:
                    circ.compose(c, inplace=True)

        return circ



    def construct_trotter_circuit_old(self,
                                  dt: float,
                                  trotter: List[int],
                                  order: int,
                                  fac: float = None,
                                  init_reg_to_zero: bool = False,
                                  q: QuantumRegister = None):
        """
        Constructs circuit for Polaron variational form
        Args:
            params: (time-dependent) variational parameters f_k^s
            q: initial quantum register
            omega: boson self-energies for each mode
        """


        if q is None:
            fr = QuantumRegister(self._num_fermions, 'f')
            br = QuantumRegister(self._num_bosons*self._num_modals, 'b')
        elif len(q) != self._num_qubits:
            raise ValueError('If passed as an argument, the number of qubits '
                             'has to match requirement N_q = N_ferm + '
                             'N_bos * N_modals, in this case, '
                             '{}'.format(self._num_qubits))
        else:
            fr = q[0:self._num_fermions]
            br = q[self._num_fermions:]

        trotter_depth_global = trotter[0]
        trotter_depth_local = trotter[1]

        if order == 1:
            dt_new = dt / (trotter_depth_global)
        elif order == 2:
            dt_new = dt / (2*trotter_depth_global)
        elif order > 2:
            if fac == None:
                raise ValueError('Must supply argument fac')
            dt_new = fac*dt / (2*trotter_depth_global)

        circ = QuantumCircuit(fr, br)

# TODO: Add support for initial state OR option that initial state is handled
#       in main script

# NOTE: only initialize circuit ONCE per trotter circuit. for Trotter order > 1
# must NOT add this X-gate except for first "layer" of Trotter (first U_2k-2)
        if (q is None) and (not init_reg_to_zero):
            for k in range(self._num_bosons):
                circ.x(br[k*self._num_modals]) # initialize k-mode to |01..0>


        for r in range(trotter_depth_global):
            if self._coupling == 'x':
                circ.h(fr[0])
            elif self._coupling == 'y':
                circ.rx(-np.pi/2, fr[0])

            for k in range(self._num_bosons):
                for s in range(trotter_depth_local):
                    # append exp(... X_even) to circuit
                    for nk in range(0, self._num_modals - 1, 2):
                        gate_param = (dt_new*self._g_couple[k]*np.sqrt(nk+1)
                                      /(trotter_depth_local))

                        circ += self._build_controlled_two_qubit_gate(
                                    0, k*self._num_modals + nk, gate_param)

                    # append exp(... X_odd) to circuit
                    for nk in range(1, self._num_modals - 1, 2):
                        gate_param = (dt_new*self._g_couple[k]*np.sqrt(nk+1)
                                      /(trotter_depth_local))

                        circ += self._build_controlled_two_qubit_gate(
                                    0, k*self._num_modals + nk, gate_param)

            if self._coupling == 'x':
                circ.h(fr[0])
            elif self._coupling == 'y':
                circ.rx(np.pi/2, fr[0])

            ### Construct fermion terms of Hamiltonian exponential
            # NOTE: Gate of form exp(-i/2 param sigma)
            # Thus there is a factor 2 for delta and no factor 0.5 for epsilon
            # Delta*sig_x
            circ.rx(dt_new*2*self._delta, fr[0])
            # epsilon/2 * sig_z
            circ.rz(dt_new*self._epsilon, fr[0])

            # NOTE: Gate param has factor 1/2 instead of 1/4 due to gate
            # definition exp(-i/2 param sigma), as with delta and epsilon terms
            for k in range(self._num_bosons):
                for nk in range(self._num_modals - 1):
                    gate_param = (dt_new*self._omega[k]*(nk + 1)*0.5)
                    bos_q = k*self._num_modals + nk
                    circ.cx(br[bos_q], br[bos_q + 1])
                    circ.rz(-gate_param, br[bos_q + 1])
                    circ.cx(br[bos_q], br[bos_q + 1])
                    circ.rz(gate_param, br[bos_q])
                    circ.rz(-gate_param, br[bos_q + 1])

        return circ
