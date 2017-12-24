
def convert_tuple_to_str(nametuple):
    return "".join(("["+str(i)+"]" if not isinstance(i, str) else i for i in nametuple))


def gate_qubit_tuples_of_circuit_as_one(circuit):
    N = circuit.multi_graph
    collect_all = []
    qubit_inits = {}
    cbit_inits = {}
    for n, nodedata in N.nodes_iter(data=True):
        # node is essentially a gate or an in/out node
        # print nodedata # same as N.node[n]

        nodedict = N.node[n]
        # print nodedict
        if nodedict['type'] == 'op':
            gate_args_list = []
            gatename = nodedict['name'].upper() # upper case looks better
            gateparas = nodedict['params']
            if len(gateparas) != 0:
                gatename = gatename + '(' + ",".join(gateparas) +')'

            gate_args_list.append(gatename)
            qargs = nodedict['qargs']
            for qarg in qargs:
                qubitStr = convert_tuple_to_str(qarg)
                gate_args_list.append(qubitStr)
                qubit_inits[qubitStr] = qubitStr + ' =0' # there are no other initial values than 0s
            gate_args_tuple = tuple(gate_args_list)

            cargs = nodedict['cargs']
            if cargs is not None:
                for carg in cargs:
                    cargStr = convert_tuple_to_str(carg)
                    cbit_inits[cargStr] = cargStr + ' =0' # there are no other initial values than 0s

            collect_all.append(gate_args_tuple)

        elif nodedict['type'] == 'in':
            continue
        elif nodedict['type'] == 'out':
            continue
        else:
            return NotImplemented
    return collect_all, qubit_inits, cbit_inits



def gate_qubit_tuples_of_circuit_as_layers(circuit):
    layers = circuit.layers() #another option is: serial_layers
    layer_id = -1
    schedule_list_of_layers = []
    global_qubit_inits = {}
    global_cbit_inits = {}

    for layer in layers: # each layer is essentially implemented with a circuit
        layer_id = layer_id + 1
        oneLayer = layer.get('graph')
        list_per_layer, qubit_inits_per_layer, cbit_inits_per_layer = gate_qubit_tuples_of_circuit_as_one(oneLayer) # each layer is a circuit
        # list_per_layer: [('H', 'var0'), ('H', 'var1'), ('H', 'var2'), ('X', 'conj0'), ('X', 'conj1'), ('X', 'conj2')]

        schedule_list_of_layers.append(list_per_layer)
        global_qubit_inits.update(qubit_inits_per_layer)
        global_cbit_inits.update(cbit_inits_per_layer)
    return schedule_list_of_layers, global_qubit_inits, global_cbit_inits

