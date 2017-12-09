import pydot
from circuit_layer_analyzer import convert_tuple_to_str
from circuit_layer_analyzer import gate_qubit_tuples_of_circuit_as_layers, gate_qubit_tuples_of_circuit_as_one
from plot_quantum_circuit import plot_quantum_circuit
from plot_quantum_circuit import plot_quantum_schedule, save_plot



def dotGetNodeLabel(N, nid): # what the user will see on the graph
    nodedict = N.node[nid]
    if nodedict['type'] == 'op':
        return nodedict['name']
    elif nodedict['type'] == 'in':
        if isinstance(nodedict['name'], tuple):
            return convert_tuple_to_str(nodedict['name'])
        else:
            return NotImplemented
    elif nodedict['type'] == 'out':
        if isinstance(nodedict['name'], tuple):
            return convert_tuple_to_str(nodedict['name'])
        else:
            return NotImplemented
    else:
        return NotImplemented



def dotGetNodeName(N, nid): # used internally by dot to distinguish nodes.
    return dotGetNodeLabel(N, nid) + str(nid)


def dot(circuit, png_name = 'circuit.png'):
    N = circuit.multi_graph

    if not N.is_multigraph() or not N.is_directed():
        return

    P = pydot.Dot(graph_type='digraph', strict=False)
    # nodes:
    for n, nodedata in N.nodes_iter(data=True):
        # node is essentially a gate or an in/out node
        # print nodedata # same as N.node[n]
        p=pydot.Node(dotGetNodeName(N, n), label = dotGetNodeLabel(N, n))
        P.add_node(p)
    # edges:
    for u,v,key,edgedata in N.edges_iter(data=True,keys=True):
        # edge is essentially qubit, e.g.,  q[i]
        # print edgedata
        edgeLabel = convert_qubit_tuple_to_str(edgedata['name'])
        edge=pydot.Edge(src=dotGetNodeName(N, u), dst=dotGetNodeName(N, v), label=edgeLabel)
        P.add_edge(edge)
    # output:
    P.write_png(png_name)

    # the following method is convenient but not flexible:
    # nx.draw(N)
    # plt.show()


def circuit_as_one(circuit, png_name ='circuit.png'):
    N = circuit.multi_graph
    if not N.is_multigraph() or not N.is_directed():
        return

    # you only need to provide "(gate, qubit)" tuples, the underlying plotter will figure out everything
    # examples of each tuple: ('S','j_0','j_1'), ('H','j_0')
    # full example: plot_quantum_circuit([('H','j_0'),('S','j_0','j_1'),('T','j_0','j_2'),('H','j_1'), ('S','j_1','j_2'),('H','j_2'),('SWAP','j_0','j_2')])
    collect_all, qubit_inits = gate_qubit_tuples_of_circuit_as_one(circuit)
    plot_quantum_circuit(collect_all, inits=qubit_inits)


def save_plot_only(mycircuit, png_path ='circuit.png'):
    schedule_list_of_layers, global_qubit_inits, _ = gate_qubit_tuples_of_circuit_as_layers(mycircuit)
    save_plot(png_path, schedule_list_of_layers, inits=global_qubit_inits)

#plot_quantum_schedule([[('H','q0')], [('CNOT','q1','q0')]])
def circuit_as_layers(circuit, png_name ='circuit.png'):
    schedule_list_of_layers, global_qubit_inits, _ = gate_qubit_tuples_of_circuit_as_layers(circuit)
    plot_quantum_schedule(schedule_list_of_layers, inits=global_qubit_inits)




def visualize(args, circuit):
    if hasattr(circuit, 'multi_graph'):

        if args.visualize_dot:
            dot(circuit)
        elif args.visualize_circuit:
            circuit_as_one(circuit)
        elif args.visualize_circuit_layers:
            circuit_as_layers(circuit)







        # if args.graph_analysis_print_nodes:
        #     ids = g.nodes()
        #     for nid in ids:
        #         print g.node[nid]

        # self.multi_graph.node[self.node_counter]["type"] = "op"
        # self.multi_graph.node[self.node_counter]["name"] = nname
        # self.multi_graph.node[self.node_counter]["qargs"] = nqargs
        # self.multi_graph.node[self.node_counter]["cargs"] = ncargs
        # self.multi_graph.node[self.node_counter]["params"] = nparams
        # self.multi_graph.node[self.node_counter]["condition"] = ncondition
