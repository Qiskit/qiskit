# from qiskit.aqua.components.variational_forms import VariationalForm
from qiskit import QuantumCircuit, QuantumRegister

import numpy as np
from typing import Optional, Union, List
import warnings


class SBHamil():
    """This trial wave function is a modification of the Polaron variational
    form. It includes additional fermionic terms in the exponential and two
    separate Trotter-expansions; one splitting fermionic and polaron terms into
    exp[fermion terms] * exp[polaron] ('global Trotter'), and one splitting the
    polaron itself ('local Trotter').
    The class contains methods to construct the quantum circuit for the
    variational form, but also for derivatives thereof with respect to
    variational parameters.
    The modifications to the polaron form stem from the need to include
    tunneling of the fermion between its two states (introduced by a pauli-x
    term in the Hamiltonian), which the original polaron was unable to account
    for.
    The polaron ansatz is designed for applications to spin-boson models. It
    was derived for quantum circuits in
    http://arxiv.org/abs/1909.08640
    For a theoretical investigation, see, e.g.,
    https://arxiv.org/abs/1512.04244
    https://arxiv.org/abs/1701.04709

    A direct boson-to-qubit mapping (SES encoding) is used
    """

    def __init__(self,
                 num_particles: Union[List[int], int],
                 num_modals: int,
                 trotter_depth: Union[List[int], int],
                 coupling: str = 'x',
                 modal_params_distinct: bool = True,
                 adjoint_params_distinct: bool = False):
        """Constructor.

        Args:
            num_particles: number of particles, if (list[int]), first entry
                           gives number of fermions, second entry number of
                           bosons. If (int), number of fermions assumed to be
                           1, number of bosons is num_particles - 1
            num_modals: gives number of modals per boson (ground state
                        + excited states)
            trotter_depth: number of Trotter steps
        """

        # super().__init__()

        if isinstance(num_particles, list):
            if num_particles[0] != 1:
                warnings.warn('Currently only 1 Fermion is supported! '
                              'Setting fermion number to 1')
            self._num_fermions = 1
            self._num_bosons = num_particles[1]
        else:
            self._num_fermions = 1
            self._num_bosons = num_particles-1

        self._num_modals = num_modals
        self._num_ancilla = 0

        self._num_qubits = (self._num_fermions
                            + self._num_modals*self._num_bosons)

        if (coupling == 'x') or (coupling == 'y') or (coupling == 'z'):
            self._coupling = coupling
        else:
            raise ValueError('If supplied as an argument, coupling must be '
                             'of type string and either be "x", "y", or "z"')

        if not isinstance(trotter_depth, list):
            self._trotter_depth_global = trotter_depth
            self._trotter_depth_local = 1
        else:
            self._trotter_depth_global = trotter_depth[0]
            self._trotter_depth_local = trotter_depth[1]

        self._modal_params_distinct = modal_params_distinct
        self._adjoint_params_distinct = adjoint_params_distinct

        if not self._modal_params_distinct:
            self._num_parameters = (self._num_bosons
                                    *self._trotter_depth_local)
        else:
            self._num_parameters = (self._num_bosons
                                    *self._trotter_depth_local
                                    *(self._num_modals - 1))

        if self._adjoint_params_distinct:
            self._num_parameters *= 2

        if self._modal_params_distinct:
            self._num_parameters += self._num_bosons*(self._num_modals - 1)
        else:
            self._num_parameters += self._num_bosons

        self._num_parameters = (self._trotter_depth_global
                                *(self._num_parameters + 2))

    @property
    def num_parameters(self):
        return self._num_parameters

    @property
    def num_qubits(self):
        return self._num_qubits


    def _build_controlled_two_qubit_gate(self, ctrl_q, tar_q, param, param2):
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

        assert ((tar_q+1) % self._num_modals != 0), ('First target of two-'
            'qubit gate cannot be last qubit in k-mode (k-th boson) register')

        fr = QuantumRegister(self._num_fermions, 'f')
        br = QuantumRegister(self._num_bosons*self._num_modals, 'b')
        twoQgate = QuantumCircuit(fr, br)

        # TODO: rather use U1 gate instead of Rz?
        twoQgate.h(br[tar_q]) # rotates Z to X basis
        twoQgate.h(br[tar_q + 1])
        twoQgate.cx([fr[ctrl_q], br[tar_q]], [br[tar_q], br[tar_q+1]])
        twoQgate.rz(param, br[tar_q + 1])
        twoQgate.cx([br[tar_q], fr[ctrl_q]], [br[tar_q+1], br[tar_q]])
        twoQgate.h(br[tar_q + 1])
        twoQgate.h(br[tar_q]) # rotates X to Z basis

        twoQgate.rx(-np.pi/2, br[tar_q]) # rotates Z to Y basis
        twoQgate.rx(-np.pi/2, br[tar_q + 1])
        twoQgate.cx([fr[ctrl_q], br[tar_q]], [br[tar_q], br[tar_q+1]])
        twoQgate.rz(param2, br[tar_q + 1])
        twoQgate.cx([br[tar_q], fr[ctrl_q]], [br[tar_q+1], br[tar_q]])
        twoQgate.rx(np.pi/2, br[tar_q + 1])
        twoQgate.rx(np.pi/2, br[tar_q]) # rotates Y to Z basis

        return twoQgate



    def construct_circuit(self, params, q=None):
        """
        Constructs circuit for Polaron variational form
        Args:
            params: (time-dependent) variational parameters f_k^s
            q: initial quantum register
        """

        if len(params) != self._num_parameters:
            raise ValueError('The number of parameters passed ({}) has to '
                             'match the requirement N_params = N_bosons * '
                             'N_TrotterSteps (default argument '
                             'trotter_depth = 2), in this case, '
                             '{}'.format(len(params), self._num_parameters))

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
        param_count = 0

# TODO: Add support for initial state OR option that initial state is handled
#       in main script
        if q is None:
            for k in range(self._num_bosons):
                circ.x(br[k*self._num_modals]) # initialize k-mode to |01..0>


        for r in range(self._trotter_depth_global):
            if self._coupling == 'x':
                circ.h(fr[0])
            elif self._coupling == 'y':
                circ.rx(-np.pi/2, fr[0])

            for k in range(self._num_bosons):
                for s in range(self._trotter_depth_local):
                    # append exp(... X_even) to circuit
                    for nk in range(self._num_modals - 1):
                        if (nk % 2 == 0):
                            gate_param_fac = (np.sqrt(nk+1)
                                          /(self._trotter_depth_local
                                            *self._trotter_depth_global))
                            gate_param = params[param_count]*gate_param_fac
                            if not self._adjoint_params_distinct:
                                gate_param2 = (params[param_count]
                                               *gate_param_fac)
                            else:
                                gate_param2 = (params[param_count + 1]
                                               *gate_param_fac)

                            circ += self._build_controlled_two_qubit_gate(
                                0, k*self._num_modals + nk,
                                gate_param, gate_param2)

                            if (self._modal_params_distinct
                                    and self._adjoint_params_distinct):
                                param_count += 2
                            elif (self._modal_params_distinct
                                    and not self._adjoint_params_distinct):
                                param_count += 1

                    # append exp(... X_odd) to circuit
                    for nk in range(self._num_modals - 1):
                        if (nk % 2 != 0):
                            gate_param_fac = (np.sqrt(nk+1)
                                          /(self._trotter_depth_local
                                            *self._trotter_depth_global))
                            gate_param = params[param_count]*gate_param_fac
                            if not self._adjoint_params_distinct:
                                gate_param2 = (params[param_count]
                                               *gate_param_fac)
                            else:
                                gate_param2 = (params[param_count + 1]
                                               *gate_param_fac)

                            circ += self._build_controlled_two_qubit_gate(
                                0, k*self._num_modals + nk,
                                gate_param, gate_param2)

                            if (self._modal_params_distinct
                                    and self._adjoint_params_distinct):
                                param_count += 2
                            elif (self._modal_params_distinct
                                    and not self._adjoint_params_distinct):
                                param_count += 1

                    if (not self._modal_params_distinct
                            and not self._adjoint_params_distinct):
                        param_count += 1
                    elif (self._adjoint_params_distinct
                            and not self._modal_params_distinct):
                        param_count += 2

            if self._coupling == 'x':
                circ.h(fr[0])
            elif self._coupling == 'y':
                circ.rx(np.pi/2, fr[0])

            ### Construct fermion terms of Hamiltonian exponential
            # Delta*sig_x
            circ.rx(params[param_count]*2/self._trotter_depth_global, fr[0])
            param_count += 1
            # epsilon/2 * sig_z
            circ.rz(params[param_count]/self._trotter_depth_global, fr[0])
            param_count += 1

            for k in range(self._num_bosons):
                for nk in range(self._num_modals - 1):
                    gate_param = 0.5*params[param_count]*(nk + 1)
                    bos_q = k*self._num_modals + nk
                    circ.cx(br[bos_q], br[bos_q + 1])
                    circ.rz(-gate_param, br[bos_q + 1])
                    circ.cx(br[bos_q], br[bos_q + 1])
                    circ.rz(gate_param, br[bos_q])
                    circ.rz(-gate_param, br[bos_q + 1])

                    if self._modal_params_distinct:
                        param_count += 1
                if not self._modal_params_distinct:
                    param_count += 1

        return circ
