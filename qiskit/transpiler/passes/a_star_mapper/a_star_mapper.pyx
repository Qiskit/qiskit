from libcpp.queue cimport queue, priority_queue
from libcpp.set cimport set
from libcpp.pair cimport pair
from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector
from libc.string cimport memcpy
from libcpp.utility cimport pair
from libcpp cimport bool
from libc.stdio cimport *
from libc.limits cimport INT_MAX
from libc.stdint cimport uintptr_t
cimport cython

# breadth-first search algorithm to find minimal distances between physical qubits
cdef int bfs(int start, int** dist, set[pair[int, int] ]& coupling_graph):
    cdef queue[int] q
    cdef set[int] visited
    cdef int v
    cdef pair[int, int] edge

    visited.insert(start)
    q.push(start)
    dist[start][start] = 0

    while not q.empty(): 
        v = q.front()
        q.pop();
        for edge in coupling_graph:
            if edge.first == v and visited.find(edge.second) == visited.end():
                visited.insert(edge.second)
                q.push(edge.second)
                dist[start][edge.second] = dist[start][v] + 1
            elif edge.second == v and visited.find(edge.first) == visited.end():
                visited.insert(edge.first)
                q.push(edge.first)
                dist[start][edge.first] = dist[start][v] + 1


# define class for nodes in the A* search
cdef cppclass a_star_node_mapper:
    int cost_fixed # fixed cost of the current permutation
    int cost_heur  # heuristic cost of the current permutation
    int *locations # location (i.e. pysical qubit) of a logical qubit
    int *qubits    # logical qubits that are mapped to the physical ones 
    bool is_goal   # true if the node is a goal node
    vector[pair[int, int]] swaps # a sequence of swap operations that have been applied 

# A* search algorithm to find a sequence of swap gates such that at least one gate in gates can be applied
@cython.boundscheck(False)
@cython.wraparound(False)
cdef a_star_node_mapper* a_star_search(set[pair[int, int]]& gates, int* map, int* loc, int** dist, int nqubits, set[pair[int, int] ]& coupling_graph, set[pair[int, int]]& free_swaps) except + :
    cdef priority_queue[pair[int, uintptr_t]] q
    cdef a_star_node_mapper* current
    cdef a_star_node_mapper* new_node
    cdef int tmp_qubit1, tmp_qubit2
    cdef pair[int, int] g, edge
    cdef set[int] used_qubits
    cdef set[int] interacted_qubits

    # determine all qubits that occur in a 2-qubit gate that can be applied
    for g in gates:
        used_qubits.insert(g.first)
        used_qubits.insert(g.second)

    # create a new node representing the current mapping
    current = new a_star_node_mapper()
    current.cost_fixed = 0
    current.cost_heur = 0
    current.qubits = <int*>malloc(nqubits * sizeof(int))
    current.locations = <int*>malloc(nqubits * sizeof(int))    
    memcpy(current.qubits, map, sizeof(int) * nqubits)
    memcpy(current.locations, loc, sizeof(int) * nqubits)
    current.is_goal = False
    current.swaps = vector[pair[int, int]]()

    q.push(pair[int, uintptr_t](current.cost_fixed + current.cost_heur, <uintptr_t>current))

    # perform A* search
    while not (<a_star_node_mapper*>q.top().second).is_goal:
        current = <a_star_node_mapper*> (q.top().second)
        q.pop()

        # determine all successor nodes (one for each SWAP gate that can be applied)
        for edge in coupling_graph:
            # apply only SWAP operations including at least one qubit in used_qubits 
            if used_qubits.find(current.qubits[edge.first]) == used_qubits.end() and used_qubits.find(current.qubits[edge.second]) == used_qubits.end():
                continue
            # do not apply the same SWAP gate twice in a row
            if current.swaps.size() > 0:
                g = current.swaps[current.swaps.size()-1] 
                if g.first == edge.first and g.second == edge.second:
                    continue

            # create a new node
            new_node = new a_star_node_mapper()
            new_node.qubits = <int*>malloc(nqubits * sizeof(int))
            new_node.locations = <int*>malloc(nqubits * sizeof(int))    
            new_node.swaps = current.swaps
            new_node.swaps.push_back(edge)

            # initialize the new node with the mapping of the current node 
            memcpy(new_node.qubits, current.qubits, sizeof(int) * nqubits)
            memcpy(new_node.locations, current.locations, sizeof(int) * nqubits)
      
            # update mapping of the qubits resulting from adding a SWAP gate
            tmp_qubit1 = new_node.qubits[edge.first]
            tmp_qubit2 = new_node.qubits[edge.second]
            new_node.qubits[edge.first] = tmp_qubit2
            new_node.qubits[edge.second] = tmp_qubit1

            if tmp_qubit1 != -1:
                new_node.locations[tmp_qubit1] = edge.second
            if tmp_qubit2 != -1:
                new_node.locations[tmp_qubit2] = edge.first

            # determine fixed cost of new node
            interacted_qubits.clear()
            new_node.cost_fixed = 0
            for edge in new_node.swaps:
                # only add the cost of a swap gate if it is not "free"
                if interacted_qubits.find(edge.first) != interacted_qubits.end() or interacted_qubits.find(edge.first) != interacted_qubits.end() or free_swaps.find(edge) == free_swaps.end():
                    new_node.cost_fixed += 1
                interacted_qubits.insert(edge.first)
                interacted_qubits.insert(edge.second)

            new_node.is_goal = False 
            new_node.cost_heur = 0 

            # Check wheter a goal state is reached (i.e. whether any gate can be applied) and determine heuristic cost
            for g in gates:
                if dist[new_node.locations[g.first]][new_node.locations[g.second]] == 1:
                    new_node.is_goal = True
                # estimate remaining cost (the heuristic is not necessarily admissible and, hence, may yield to sub-optimal local solutions) 
                new_node.cost_heur += dist[new_node.locations[g.first]][new_node.locations[g.second]] - 1

            # add new node to the queue
            q.push(pair[int, uintptr_t](INT_MAX - (new_node.cost_fixed + new_node.cost_heur), <uintptr_t>new_node))

        # delete current node
        free(current.locations)
        free(current.qubits)
        del(current)
    
    current = <a_star_node_mapper*>(q.top().second)
    q.pop()

    # clean up
    while not q.empty():
        new_node = <a_star_node_mapper*>(q.top().second)
        free(new_node.locations)
        free(new_node.qubits)
        del(new_node)
        q.pop()

    return current
        
# function to rewrite gates with the current mapping and add them to the compiled circuit
cdef add_rewritten_gates(gates_original, int* locations, compiled_circuit):
    qubit_names = compiled_circuit.get_qubits()    
    for g in gates_original:
        qargs_new = []
        for qarg in g['qargs']:
            qargs_new += [('q', locations[qubit_names.index(qarg)])]
        g['qargs'] = qargs_new
        compiled_circuit.apply_operation_back(g['name'], qargs_new, g['cargs'], g['params'], g['condition'])
    return compiled_circuit            


# define class for nodes in the A* search
cdef cppclass a_star_node_initial_mapping:
    int cost_fixed # fixed cost of the current permutation
    int cost_heur  # heuristic cost of the current permutation
    int *locations # location (i.e. pysical qubit) of a logical qubit
    int *qubits    # logical qubits that are mapped to the physical ones
    vector[pair[int, int]] remaining_gates # vector holding the initial gates that have to be mapped to an edge of the coupling map

@cython.boundscheck(False)
@cython.wraparound(False)
cdef a_star_node_initial_mapping* find_initial_permutation(int nqubits, set[pair[int, int]] initial_gates, int* first_interaction, int** dist, set[pair[int, int]] coupling_graph):
    cdef priority_queue[pair[int, uintptr_t]] q
    cdef a_star_node_initial_mapping* current
    cdef a_star_node_initial_mapping* new_node
    cdef int i, j, k, min_dist
    cdef bool mappingOK
    cdef pair[int, int] gate, edge

    # create a new node representing the initial mapping (none of the qubits is mapped yet)
    current = new a_star_node_initial_mapping()
    current.cost_fixed = 0
    current.cost_heur = 0
    current.locations = <int*>malloc(nqubits * sizeof(int))
    current.qubits = <int*>malloc(nqubits * sizeof(int))
    for i in range(nqubits):
        current.locations[i] = -1
        current.qubits[i] = -1
    for gate in initial_gates:
        current.remaining_gates.push_back(gate)
    
    q.push(pair[int, uintptr_t](current.cost_fixed + current.cost_heur, <uintptr_t>current))

    # perform A* search
    while (<a_star_node_initial_mapping*>q.top().second).remaining_gates.size() != 0:
        current = <a_star_node_initial_mapping*> (q.top().second)
        q.pop()
        gate = current.remaining_gates.back()
        current.remaining_gates.pop_back()

        # determine all successor nodes (a gate group acting on a pair of qubits can be applied to any edge in the coupling map)
        # we enforce mapping these groups to an edge in the coupling map in order to avoid SWAPs before appliying the first gate
        for edge in coupling_graph:
            if current.qubits[edge.first] != -1 or current.qubits[edge.second] != -1:
                continue

            # create a new node
            new_node = new a_star_node_initial_mapping()
            new_node.locations = <int*>malloc(nqubits * sizeof(int))    
            new_node.qubits = <int*>malloc(nqubits * sizeof(int))    

            # initialize the new node with the mapping of the current node 
            memcpy(new_node.locations, current.locations, sizeof(int) * nqubits)
            memcpy(new_node.qubits, current.qubits, sizeof(int) * nqubits)

            new_node.remaining_gates = current.remaining_gates

            new_node.qubits[edge.first] = gate.first       
            new_node.locations[gate.first] = edge.first

            new_node.qubits[edge.second] = gate.second       
            new_node.locations[gate.second] = edge.second
            
            new_node.cost_fixed = current.cost_fixed
            new_node.cost_heur = 0
            if first_interaction[gate.first] != -1 and new_node.locations[first_interaction[gate.first]] != -1:
                new_node.cost_fixed += dist[new_node.locations[gate.first]][new_node.locations[first_interaction[gate.first]]]
            else:
                min_dist = nqubits
                for k in range(0, nqubits):
                    if new_node.qubits[k] == -1:
                        min_dist = min(min_dist, dist[new_node.locations[gate.first]][k])
                new_node.cost_heur += min_dist
            if first_interaction[gate.second] != -1 and new_node.locations[first_interaction[gate.second]] != -1:
                new_node.cost_fixed += dist[new_node.locations[gate.second]][new_node.locations[first_interaction[gate.second]]]
            else:
                min_dist = nqubits
                for k in range(0, nqubits):
                    if new_node.qubits[k] == -1:
                        min_dist = min(min_dist, dist[new_node.locations[gate.second]][k])
                new_node.cost_heur += min_dist

            q.push(pair[int, uintptr_t](INT_MAX - (new_node.cost_fixed + new_node.cost_heur), <uintptr_t>new_node))


            # create a second new node (since there are two qubits involved) 
            new_node = new a_star_node_initial_mapping()
            new_node.locations = <int*>malloc(nqubits * sizeof(int))    
            new_node.qubits = <int*>malloc(nqubits * sizeof(int))    

            # initialize the new node with the mapping of the current node 
            memcpy(new_node.locations, current.locations, sizeof(int) * nqubits)
            memcpy(new_node.qubits, current.qubits, sizeof(int) * nqubits)

            new_node.remaining_gates = current.remaining_gates

            new_node.qubits[edge.second] = gate.first       
            new_node.locations[gate.first] = edge.second

            new_node.qubits[edge.first] = gate.second       
            new_node.locations[gate.second] = edge.first
            
            new_node.cost_fixed = current.cost_fixed
            new_node.cost_heur = 0
            if first_interaction[gate.first] != -1 and new_node.locations[first_interaction[gate.first]] != -1:
                new_node.cost_fixed += dist[new_node.locations[gate.first]][new_node.locations[first_interaction[gate.first]]]
            else:
                min_dist = nqubits
                for k in range(0, nqubits):
                    if new_node.qubits[k] == -1:
                        min_dist = min(min_dist, dist[new_node.locations[gate.first]][k])
                new_node.cost_heur += min_dist
            if first_interaction[gate.second] != -1 and new_node.locations[first_interaction[gate.second]] != -1:
                new_node.cost_fixed += dist[new_node.locations[gate.second]][new_node.locations[first_interaction[gate.second]]]
            else:
                min_dist = nqubits
                for k in range(0, nqubits):
                    if new_node.qubits[k] == -1:
                        min_dist = min(min_dist, dist[new_node.locations[gate.second]][k])
                new_node.cost_heur += min_dist

            q.push(pair[int, uintptr_t](INT_MAX - (new_node.cost_fixed + new_node.cost_heur*<int>1), <uintptr_t>new_node))    

        # delete current node  
        free(current.locations)
        free(current.qubits)
        del(current)
           
    current = <a_star_node_initial_mapping*>(q.top().second)
    q.pop()

    # clean up
    while not q.empty():
        new_node = <a_star_node_initial_mapping*>(q.top().second)
        free(new_node.locations)
        free(new_node.qubits)
        del(new_node)
        q.pop()
    return current
    

# main method for performing the mapping algorithm
@cython.boundscheck(False)
@cython.wraparound(False)
def a_star_mapper(grouped_gates, coupling_map, int nqubits, empty_circuit):
    compiled_circuit = empty_circuit
 
    # Switch to C/C++ for performance reasons

    cdef int** dist = <int**>malloc(nqubits * sizeof(int*))
    cdef int i,j
    cdef vector[pair[int, int]] applied_gates

    # allocate 2-dimensional array for the distance between two physical qubits
    for i in range(nqubits):
        dist[i] = <int*>malloc(nqubits * sizeof(int))

    # translate the coupling_map to a C++ set of pairs
    cdef set[pair[int, int] ] coupling_graph    
    for key, value in coupling_map.items():
        for v in value:
            coupling_graph.insert((key, v))

    # determine the minimal distances between two qubits    
    for i in range(nqubits):
        bfs(i,dist,coupling_graph)

    
    # allocate arrays for the mapping of the qubits
    # locations[q1] for a logical qubit q1 gives the physical qubit that q1 is mapped to
    # qubits[Q1] for a physical qubit Q1 gives the logaic qubit that is mapped to Q1 
    cdef int *locations = <int*>malloc(nqubits * sizeof(int))
    cdef int *qubits = <int*>malloc(nqubits * sizeof(int))
    # start with a mapping that is initially empty (none of the logical qubits is mapped to a physical one)
    for i in range(0, nqubits):
        locations[i] = -1
        qubits[i] = -1

    cdef int q0, q1, ii, iii

    cdef a_star_node_initial_mapping* init_perm
    cdef set[pair[int, int]] initial_gates
    cdef int* first_interaction
    # search for "best initial mapping" (regarding a certain heuristic) using an A* search algorithm. This is only feasable for a small number of qubits
    if nqubits <= 8:
        first_interaction = <int*>malloc(nqubits * sizeof(int))

        for i in range(0, nqubits):
            first_interaction[i] = -1

        for node in grouped_gates.nodes:
            degree = grouped_gates.in_degree(node)
            if degree == 0:
                if len(grouped_gates.node[node]['qubits']) == 2:
                    q0 = grouped_gates.node[node]['qubits'][0]
                    q1 = grouped_gates.node[node]['qubits'][1]
                    # the mapping for all gates in the first layer shall be satified
                    initial_gates.insert(pair[int, int](q0, q1))
                    # determine the qubit with which q0 and q1 interact next  
                    for succ in grouped_gates.successors(node):
                        if q0 == grouped_gates.node[succ]['qubits'][0]:
                            first_interaction[grouped_gates.node[succ]['qubits'][1]] = q0
                        elif q0 == grouped_gates.node[succ]['qubits'][1]:
                            first_interaction[grouped_gates.node[succ]['qubits'][0]] = q0
                        elif q1 == grouped_gates.node[succ]['qubits'][0]:
                            first_interaction[grouped_gates.node[succ]['qubits'][1]] = q1
                        elif q1 == grouped_gates.node[succ]['qubits'][1]:
                            first_interaction[grouped_gates.node[succ]['qubits'][0]] = q1

        # call an A* algorithm to determe the best initial mapping    
        init_perm = find_initial_permutation(nqubits, initial_gates, first_interaction, dist, coupling_graph)

        free(locations)
        locations = init_perm.locations
        free(qubits)
        qubits = init_perm.qubits
        free(first_interaction)
        del(init_perm)

    cdef set[pair[int, int]] applicable_gates
    cdef a_star_node_mapper* result

    cdef set[int] used_qubits
    cdef set[pair[int,int]] free_swaps

    # conduct the mapping of the circuit
    while grouped_gates.order() > 0:
        # add all gates that can be directly applied to the circuit
        while True:
            applicable_gates.clear()
            nodes_to_remove = []
            for node in grouped_gates.nodes:
                degree = grouped_gates.in_degree(node)
                if degree == 0:
                    if len(grouped_gates.node[node]['qubits']) != 2:
                        # add measurement and barrier gates to the compiled circuit
                        for q in grouped_gates.node[node]['qubits']:
                            for qq in range(nqubits):
                                if locations[qq] == -1:
                                    locations[qq] = q
                                    qubits[q] = qq
                                    break
                        compiled_circuit = add_rewritten_gates(grouped_gates.node[node]['gates'], locations, compiled_circuit)
                        nodes_to_remove += [node]
                    else:
                        # map all yet unmapped qubits that occur in gates that can be applied

                        q0 = grouped_gates.node[node]['qubits'][0]
                        q1 = grouped_gates.node[node]['qubits'][1]

                        if locations[q0] == -1 and locations[q1] == -1:
                            # case: both qubits are not yet mapped
                            min_dist = nqubits
                            min_q1 = -1
                            min_q2 = -1
                            # find best initial mapping 
                            for ii in range(nqubits):
                                for iii in range(ii+1, nqubits):
                                    if qubits[ii] == -1 and qubits[iii] == -1 and dist[ii][iii] < min_dist:
                                        min_dist = dist[ii][iii]
                                        min_q1 = ii
                                        min_q2 = iii
                            locations[q0] = min_q1
                            locations[q1] = min_q2
                            qubits[min_q1] = q0
                            qubits[min_q2] = q1                                        
                        elif locations[q0] == -1:
                            # case: only q0 is not yet mapped 
                            min_dist = nqubits
                            min_q1 = -1
                            # find best initial mapping 
                            for ii in range(nqubits):
                                if qubits[ii] == -1 and dist[ii][locations[q1]] < min_dist:
                                    min_dist = dist[ii][locations[q1]]
                                    min_q1 = ii
                            locations[q0] = min_q1
                            qubits[min_q1] = q0
                        elif locations[q1] == -1:
                            # case: only q1 is not yet mapped 
                            min_dist = nqubits
                            min_q1 = -1
                            # find best initial mapping 
                            for ii in range(nqubits):
                                if qubits[ii] == -1 and dist[ii][locations[q0]] < min_dist:
                                    min_dist = dist[ii][locations[q0]]
                                    min_q1 = ii
                            locations[q1] = min_q1
                            qubits[min_q1] = q1

                        # gates with a distance of 1 can be directly applied
                        if dist[locations[q0]][locations[q1]] == 1:
                            compiled_circuit = add_rewritten_gates(grouped_gates.node[node]['gates'], locations, compiled_circuit)
                            if coupling_graph.find(pair[int, int](locations[q0], locations[q1])) != coupling_graph.end():
                                applied_gates.push_back(pair[int, int](locations[q0], locations[q1]))
                            else:
                                applied_gates.push_back(pair[int, int](locations[q1], locations[q0]))

                            # remove nodes representing the added gates
                            nodes_to_remove += [node]
                        else:
                            # gates with a distance greater than 1 can potentially be applied (after fixing the mapping)
                            applicable_gates.insert((q0, q1))
            if len(nodes_to_remove) == 0:
                break
            else:
                grouped_gates.remove_nodes_from(nodes_to_remove)
        
        # check whether all gates have been successfully applied 
        if len(applicable_gates) == 0:
            break

        # determine which SWAPs can be applied for "free". A SWAP on qubits q0 and q1 does not cost anything if the group 
        # of gates between q0 and q1 have been directly applied before it. This assumption is justified by the post-processing 
        # we use and can be included in the heuristic of the search algorithm.
        used_qubits.clear()
        free_swaps.clear()
        for i in range(applied_gates.size()-1,-1,-1):
            if used_qubits.find(applied_gates[i].first) == used_qubits.end() and used_qubits.find(applied_gates[i].second) == used_qubits.end():
                free_swaps.insert(applied_gates[i])
            used_qubits.insert(applied_gates[i].first)
            used_qubits.insert(applied_gates[i].second)
    

        # Apply A* to find a permutation such that further gates can be applied
        result = a_star_search(applicable_gates, qubits, locations, dist, nqubits, coupling_graph, free_swaps)

        # update current mapping
        free(locations)
        free(qubits)
        locations = result.locations
        qubits = result.qubits

        # add SWAPs to the compiled circuit to modify the current the mapping
        for swap in result.swaps:
            compiled_circuit.apply_operation_back('cx', [('q',swap.first), ('q',swap.second)])
            compiled_circuit.apply_operation_back('cx', [('q',swap.second), ('q',swap.first)])
            compiled_circuit.apply_operation_back('cx', [('q',swap.first), ('q',swap.second)])

            applied_gates.push_back(swap)


        del(result)

    # clean up
    free(locations)
    free(qubits)
    for i in range(nqubits):
        free(dist[i])
    free(dist)

    return compiled_circuit
