from qiskit import schedule

'''When you hit schedule, what happens? you have circuits, a backend, and an inst_map, the goal'''


# OPTION 1: replace inst_map:


def convert_inst_grape(map: object) -> object:
    pass


def grape_inst_map(backend: object, inst_map: object) -> object:
    new_inst_map = inst_map
    for i, instruction in enumerate(inst_map):
        new_map = convert_inst_grape(instruction)
        new_inst_map[i] = instruction
    pass


def schedule1(circuit, backend, inst_map=None):
    if not inst_map:
        inst_map = backend.defaults().inst_map
    inst_map = grape_inst_map(backend, inst_map)
    schedule(circuit, backend, inst_map)
    pass


# OPTION 2: wrap around schedule
def schedule_orig(pulse_sequence, backend):
    return schedule()
    pass


def circ_to_unitary(circuit):
    pass


def qutip_grape(unitary, backend):
    H_d, H_c = qutip_convert_ham(backend)
    pulse_seq = grape_runner(H_d, H_c)
    pass


def convert_circ_pulse(circuit: object, backend: object, qoc: object) -> object:
    unitary = circ_to_unitary(circuit)
    # hamiltonian_converted could be in qoc function or here

    #Or pass in the optimizer, instead of qoc pass in a function?
    #IT would look like this
    # pulse_seq = qoc(unitary, backend)
    if qoc == 'grape_qutip':

        pulse_seq = qutip_grape(unitary, backend)
        
    return pulse_seq 

def schedule2(circuit, backend, qoc=False):
    if qoc:
        qoc_pulse = convert_circ_pulse(circuit, backend, qoc)
        return schedule_orig(pulse_sequence=qoc_pulse, backend=backend)
