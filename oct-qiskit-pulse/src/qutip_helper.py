from collections import defaultdict

from qiskit.providers.aer.pulse.system_models.hamiltonian_model import HamiltonianModel
from qutip import Qobj, identity, tensor, sigmax, sigmay, sigmaz

from src.prefactor_parsing import prefactor_parser
from math import sqrt

import numpy as np
from src.helper import d_sum


def which_qubit_channel(control):
    # error checking?
    return int(control[1:])


def zeros_in_corners(matrix):
    assert(matrix.shape == (4, 4))
    return (sum(matrix[0][2:4]) + sum(matrix[1][2:4]) + sum(matrix[2][:2]) + sum(matrix[3][:2])) == 0


def raise_unitary(unitary):
    if unitary.shape[0] == 2:
        base_raiser = np.array([[1]])
        return d_sum(unitary, base_raiser)
    else:
        #! Only works for control gates rn
        # assert(zeros_in_corners(unitary))
        upper_left = np.array([unitary[0][:2], unitary[1][:2]])
        bottom_right = np.array([unitary[0][:2], unitary[1][:2]])
        return d_sum(d_sum(d_sum(upper_left, identity(1)), d_sum(bottom_right, identity(1))), identity(3))
        # return d_sum(d_sum(identity(3), unitary), identity(3))


def two_level_ham(config, subsystem_list, add_y, no_control):
    # TODO fix 2q version


    bigy1 = Qobj(tensor(sigmay(), identity(2)).full())
    bigy2 = Qobj(tensor(identity(2), sigmay()).full())
    bigx1 = Qobj(tensor(sigmax(), identity(2)).full())
    bigx2 = Qobj(tensor(identity(2), sigmax()).full())
    hamiltonian = {}
    hamiltonian_backend = config.hamiltonian
    hamiltonian_dict = HamiltonianModel.from_dict(
        hamiltonian_backend, subsystem_list=subsystem_list)
    hamiltonian = {'H_c': {}, 'H_d': 0}
    for i, control_field in enumerate(hamiltonian_dict._system):
        matrix = (control_field[0])
        prefactor, control = prefactor_parser(control_field[1], hamiltonian_dict._variables)
        if prefactor == 0:
            continue
        if control:
            # prefactor = prefactor * 2
            if 'D' in control:
                # print("Double check that sigx2 and  sigy2 are right")
                if len(subsystem_list) == 1:
                    hamiltonian['H_c'][control] = sigmax() * prefactor
                if add_y:
                    if len(subsystem_list) == 1:
                        hamiltonian['H_c'][control + 'y'] = sigmay() * prefactor
                    elif len(subsystem_list) == 2:
                        if control == 'D0':
                            hamiltonian['H_c'][control] = Qobj(bigx1 * prefactor)
                            hamiltonian['H_c'][control + 'y'] = Qobj(bigy1 * prefactor)
                        elif control == 'D1':
                            hamiltonian['H_c'][control] = Qobj(bigx2 * prefactor)
                            hamiltonian['H_c'][control + 'y'] = Qobj(bigy2 * prefactor)
                        else:
                            raise NotImplementedError("Only q0 and q1 rn")
                    else:
                        raise NotImplementedError("Only 1-2q operations supported")
                else:
                    raise NotImplementedError("need to use y right now")
            elif no_control:
                print('not using control channels, skippping...')
            else:
                raise NotImplementedError("No use for control channels currently")
    if len(subsystem_list) == 1:
        hamiltonian['H_d'] = identity(2) * 0
    elif len(subsystem_list) == 2:
        hamiltonian['H_d'] = identity(4) * 0
    return hamiltonian


def convert_qutip_ham(config, subsystem_list, add_y=True, no_control=True, two_level=False, omegad_mult=None):
    """Convert an IBM backend to a qutip formatted hamiltonian.

    Args:
        config (backend.configuration): Backend configuration to get the hamiltonian from 
        subsystem_list (List[int]): List of qubits which the gate will act on (so far only tested on [0] or [0,1])
        add_y (bool, optional): Whether or not to perform the RWA and get a sigmay term. Defaults to True.
        no_control (bool, optional): Whether or not to ignore the control channels. Defaults to True.
        two_level (bool, optional): Whether to use a two level model or not. Defaults to False.
        omegad_mult (bool, optional): Whether to multiply the drive terms by a factor (see the writeup). Defaults to None.

    Raises:
        NotImplementedError: Only tested with qubits 0 and 1 right now, but could just remove this error
        NotImplementedError: No 3+ qubit operations examined yet.
        NotImplementedError: Right now we aren't using the control channels.

    Returns:
        [type]: [description]
    """

    if two_level:
        return two_level_ham(config, subsystem_list, add_y, no_control)

    a = Qobj([[0, 1, 0], [0, 0, sqrt(2)], [0, 0, 0]])
    adag = Qobj([[0, 0, 0], [1, 0, 0], [0, sqrt(2), 0]])

    geny = complex(0, 1) * (adag - a)

    bigy1 = Qobj(tensor(geny, identity(3)).full())
    bigy2 = Qobj(tensor(identity(3), geny).full())
    hamiltonian_backend = config.hamiltonian
    if omegad_mult:
        for i in range(input_backend.configuration().n_qubits):
            hamiltonian_backend['vars']['omegad' + str(i)] = hamiltonian_backend['vars']['omegad' + str(i)]  * omegad_mult
    hamiltonian_dict = HamiltonianModel.from_dict(
        hamiltonian_backend, subsystem_list=subsystem_list)
    hamiltonian = {'H_c': {}, 'iH_c': {}, 'H_d': 0}
    for i, control_field in enumerate(hamiltonian_dict._system):
        matrix = (control_field[0])
        prefactor, control = prefactor_parser(control_field[1], hamiltonian_dict._variables)
        if prefactor == 0:
            continue
        if control:
            # prefactor = prefactor * 2
            if 'D' in control:
                hamiltonian['H_c'][control] = Qobj(matrix.full() * prefactor)
                if add_y:
                    if len(subsystem_list) == 1:
                        hamiltonian['H_c'][control + 'y'] = Qobj(geny * prefactor)
                    elif len(subsystem_list) == 2:
                        print("Double checkthat sigx2 and  sigy2 are right")
                        if control == 'D0':
                            hamiltonian['H_c'][control + 'y'] = Qobj(bigy1 * prefactor)
                        elif control == 'D1':
                            hamiltonian['H_c'][control + 'y'] = Qobj(bigy2 * prefactor)
                        else:
                            raise NotImplementedError("Only q0 and q1 rn")
                    else:
                        raise NotImplementedError("Only 1-2q operations supported")
            elif no_control:
                print('not using control channels, skippping...')
            else:
                raise NotImplementedError("No use for control channels currently")
        elif hamiltonian['H_d'] == 0:
            hamiltonian['H_d'] = matrix * prefactor
        else:
            hamiltonian['H_d'] += matrix * prefactor
    if hamiltonian['H_d'] != 0:
        hamiltonian['H_d'] = Qobj(hamiltonian['H_d'].full())
        if hamiltonian['H_d'] == 0:
            hamiltonian['H_d'] = Qobj(np.zeros((3, 3)))
    else:
        print("Drift hamiltonian is 0, (Likely due to missing variables in backend db), do not trust results.")
        hamiltonian['H_d'] = Qobj(np.zeros((3, 3)))

    return hamiltonian
