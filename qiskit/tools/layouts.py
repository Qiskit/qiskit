# -*- coding: utf-8 -*-

# Copyright 2017 IBM RESEARCH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""
Updates the Coupling class to include plotting and layout generation routines.

import qiskit.tools.layouts to extend the Coupling class

added Class methods are:
 get_dict  : Get a dictionary representation of the coupling graph
     plot  : Plot the current coupling graph. Generates a qubit layout if none was given
  fexport  : Export the current coupling graph and qubit layout to a json file
  fimport  : Import a coupling graph and qubit layout from a json file
  gen_map  : Generate standard coupling graph and layout

"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import time
from collections import OrderedDict
from qiskit.mapper._coupling import Coupling


def __get_dict(self):
    """Return coupling dict of the coupling graph."""
    res = {}
    for e in self.get_edges():
        if e[0][1] in res.keys():
            res[e[0][1]] += [e[1][1]]
        else:
            res[e[0][1]] = [e[1][1]]
    return res


Coupling.get_dict = __get_dict


def __plot(self, scale=1.):
    """
    Plot a coupling layout as a directed graph

    Args:
        scale (float): scales the graph to help make the plot nicer

    """
    # Use the position information in self.pos to plot the circuit
    import matplotlib.pyplot as plt
    import networkx as nx
    if len(self.qubits) > 1:
        coupling_map = self.get_dict()
        G = nx.DiGraph()
        ll = list(set([elem for nn in list(coupling_map.values())
                       for elem in nn] + list(coupling_map.keys())))
        pos = self.pos
        for qnr in ll:
            if pos is None:
                G.add_node(str(qnr))
            else:
                G.add_node(str(qnr), pos=pos[qnr])
        for qnr in coupling_map:
            for tnr in coupling_map[qnr]:
                G.add_edge(str(qnr), str(tnr), weight=2)
        if pos is None:
            try:
                pos = nx.nx_pydot.pydot_layout(G, prog="neato")
                npos = np.array([np.array([pos[ii][0], -pos[ii][1]])
                                 for ii in pos])
            except BaseException:
                pos = nx.spring_layout(G, k=0.6)
                npos = np.array([np.array(pos[ii]) for ii in pos])
        else:
            pos = nx.get_node_attributes(G, 'pos')
            npos = np.array([np.array(pos[ii]) for ii in pos])
        mymax = np.amax(np.abs(npos))
        npos = npos / mymax * scale
        jj = 0
        for ii in pos:
            pos[ii] = npos[jj]
            jj += 1
        if self.pos is None:
            self.pos = npos
        xxx = np.transpose(npos)[0]
        yyy = np.transpose(npos)[1]
        dx = max(xxx) - min(xxx)
        ddx = max([dx * 1.5, 7])
        dy = max(yyy) - min(yyy)
        ddy = max([dy * 1.5, 7])
        plt.figure(figsize=(ddx, ddy))
        nx.draw_networkx_labels(G, pos, font_size=14, font_family='sans-serif')
        nx.draw_networkx_nodes(
            G,
            pos,
            node_size=400,
            node_color='lightgreen',
            alpha=1.0)
        nx.draw_networkx_edges(
            G,
            pos,
            width=3,
            arrow=True,
            edge_color='k',
            alpha=0.7,
            arrowsize=20,
            arrowstyle='->')
        plt.axis('equal')
        plt.axis('off')
        plt.show()


Coupling.plot = __plot


def __clear(self):
    # self.qubits is dict from qubit (regname,idx) tuples to node indices
    self.qubits = OrderedDict()
    # self.index_to_qubit is a dict from node indices to qubits
    self.index_to_qubit = {}
    # self.node_counter is integer counter for labeling nodes
    self.node_counter = 0
    # self.G is the coupling digraph
    self.G = nx.DiGraph()
    # self.dist is a dict of dicts from node pairs to distances
    # it must be computed, it is the distance on the digraph
    self.dist = None
    # Add edges to the graph if the couplingdict is present
    self.pos = None


Coupling.__clear = __clear


def __linear(self, n=0, order=1.0):
    """
    Creates a linear arrangement of qubits

    Args:
        n (positive integer): number of qubits in coupling map.
        order (float between 0.0  and 1.0): orientation of the coupling.
                        For a value of 0.0 the qubits are coupled from right to left.
                        For a value of 1.0 the qubits are coupled from left to right.
                        For values 0.0 < order < 1.0 random couplings are chosen with
                        probability of left-to-right given by 'order'.
    """
    self.__clear()
    if n > 0:
        cdict = {}
        for ii in range(n - 1):
            if np.random.rand() < order:
                if cdict.get(ii) is None:
                    cdict[ii] = [ii + 1]
                else:
                    cdict[ii] = [ii + 1] + cdict.get(ii)
            else:
                if cdict.get(ii + 1) is None:
                    cdict[ii + 1] = [ii]
                else:
                    cdict[ii + 1] = [ii] + cdict.get(ii + 1)

        for v0, alist in cdict.items():
            for v1 in alist:
                regname = "q"
                self.add_edge((regname, v0), (regname, v1))

        self.name = "line_rand_" + str(n)
        self.description = "Line of " + str(n) + " qubits with random cx directions and a probability of" + str(
            order * 100) + "% for coupling from left to right."
        self.pos = [[ii / n, 0] for ii in range(n)]
        self.compute_distance()


Coupling.__linear = __linear


def __circle(self, n, order=1.0):
    """
    Creates a circular arrangement of qubits

    Args:
        n (positive integer): number of qubits in coupling map.
        order (float between 0.0  and 1.0): orientation of the coupling.
                        For a value of 0.0 the qubits are coupled CCW.
                        For a value of 1.0 the qubits are coupled CW.
                        For values 0.0 < order < 1.0 random couplings are chosen with
                        probability of CW coupling given by 'order'.
    """
    self.__clear()
    if n > 0:
        cdict = {}
        for ii in range(n):
            if np.random.rand() < order:
                if cdict.get(ii) is None:
                    cdict[ii] = [(ii + 1) % n]
                else:
                    cdict[ii] = [(ii + 1) % n] + cdict.get(ii)
            else:
                if cdict.get((ii + 1) % n) is None:
                    cdict[(ii + 1) % n] = [ii]
                else:
                    cdict[(ii + 1) % n] = [ii] + cdict.get((ii + 1) % n)

        for v0, alist in cdict.items():
            for v1 in alist:
                regname = "q"
                self.add_edge((regname, v0), (regname, v1))

        self.name = "circle_" + str(n)
        self.description = "Circle with " + str(n) + " qubits with cx in random direction and a probability of" + str(
            order * 100) + "% for coupling CW."
        self.pos = [[np.sin(ii / n * 2 * np.pi),
                     np.cos(ii / n * 2 * np.pi)] for ii in range(n)]
        self.compute_distance()


Coupling.__circle = __circle


def __rect(self, n_right, n_down, order=1.0, defects=0):
    """
    Creates a rectangular arrangement of qubits

    Args:
        n_right (positive integer): number of qubits in each row.
        n_down (positive integer): number of qubits in each column.
        order (float between 0.0  and 1.0): orientation of the coupling.
                        For a value of 0.0 the qubits are coupled from right to left and bottom to top.
                        For a value of 1.0 the qubits are coupled from left to right and top to bottom.
                        For values 0.0 < order < 1.0 random couplings are chosen with
                        probability according to the value of 'order'.
        defects (integer): number of defects to introduce in the lattice.
                        A negative number of 'defects' will attempt to remove as many random links (< = abs(defects))
                        as possible without isolating any qubit from the lattice.
                        A positive number of 'defects' will add links to the lattice randmomly until either
                        all-to-all connectivity is reached or the number of added links reaches 'defects'
    """
    self.__clear()
    if n_right > 1 and n_down > 1:
        n = n_right * n_down
        cdict = {}
        for kk in range(n_down):
            for ll in range(n_right - 1):
                ii = kk * n_right + ll
                if np.random.rand() < order:
                    if cdict.get(ii) is None:
                        cdict[ii] = [ii + 1]
                    else:
                        cdict[ii] = [ii + 1] + cdict.get(ii)
                else:
                    if cdict.get(ii + 1) is None:
                        cdict[ii + 1] = [ii]
                    else:
                        cdict[ii + 1] = [ii] + cdict.get(ii + 1)
        for kk in range(n_down - 1):
            for ll in range(n_right):
                ii = kk * n_right + ll
                if np.random.rand() < order:
                    if cdict.get(ii) is None:
                        cdict[ii] = [ii + n_right]
                    else:
                        cdict[ii] = [ii + n_right] + cdict.get(ii)
                else:
                    if cdict.get(ii + n_right) is None:
                        cdict[ii + n_right] = [ii]
                    else:
                        cdict[ii + n_right] = [ii] + cdict.get(ii + n_right)

        cdict = __add_defects(cdict, defects=defects)

        for v0, alist in cdict.items():
            for v1 in alist:
                regname = "q"
                self.add_edge((regname, v0), (regname, v1))

        self.name = str(n_right) + "x" + str(n_down) + "_lattice_" + str(n)
        self.description = "Rectangular lattice with random cx directions (Probability of" + str(
            order * 100) + "% for reverse coupling)."
        nn = max(n_down, n_right)
        self.pos = [[ll / nn, -kk / nn]
                    for kk in range(n_down) for ll in range(n_right)]
        self.compute_distance()


Coupling.__rect = __rect


def __torus(self, n_right, n_down, order=1.0, defects=0):
    """
    Creates a rectangular arrangement of qubits which is linked back at the edges of the rectangle.
    This can also be represented as a torus.

    Args:
        n_right (positive integer): number of qubits in each row.
        n_down (positive integer): number of qubits in each column.
        order (float between 0.0  and 1.0): orientation of the coupling.
                        For a value of 0.0 the qubits are coupled from right to left and bottom to top.
                        For a value of 1.0 the qubits are coupled from left to right and top to bottom.
                        For values 0.0 < order < 1.0 random couplings are chosen with
                        probability according to the value of 'order'.
        defects (integer): number of defects to introduce in the lattice.
                        A negative number of 'defects' will attempt to remove as many random links (< = abs(defects))
                        as possible without isolating any qubit from the lattice.
                        A positive number of 'defects' will add links to the lattice randmomly until either
                        all-to-all connectivity is reached or the number of added links reaches 'defects'
    """
    self.__clear()
    if n_right > 1 and n_down > 1:
        n = n_right * n_down
        cdict = {}
        for kk in range(n_down):
            for ll in range(n_right):
                ii = kk * n_right
                if np.random.rand() < order:
                    if cdict.get(ii + ll) is None:
                        cdict[ii + ll] = [ii + (ll + 1) % n_right]
                    else:
                        cdict[ii + ll] = [ii + (ll + 1) %
                                          n_right] + cdict.get(ii + ll)
                else:
                    if cdict.get(ii + (ll + 1) % n_right) is None:
                        cdict[ii + (ll + 1) % n_right] = [ii + ll]
                    else:
                        cdict[ii + (ll + 1) % n_right] = [ii + ll] + cdict.get(
                            ii + (ll + 1) % n_right)
        for kk in range(n_down):
            for ll in range(n_right):
                ii = kk * n_right + ll
                if np.random.rand() < order:
                    if cdict.get(ii) is None:
                        cdict[ii] = [(ii + n_right) % n]
                    else:
                        cdict[ii] = [(ii + n_right) % n] + cdict.get(ii)
                else:
                    if cdict.get((ii + n_right) % n) is None:
                        cdict[(ii + n_right) % n] = [ii]
                    else:
                        cdict[(ii + n_right) % n] = [ii] + \
                            cdict.get((ii + n_right) % n)

        cdict = __add_defects(cdict, defects=defects)

        for v0, alist in cdict.items():
            for v1 in alist:
                regname = "q"
                self.add_edge((regname, v0), (regname, v1))

        self.name = str(n_right) + "x" + str(n_down) + "_torus_" + str(n)
        self.description = "Torus lattice with random cx directions (Probability of" + str(
            order * 100) + "% for reverse coupling)."
        nn = max(n_down, n_right)
        self.pos = [[ll / nn, -kk / nn]
                    for kk in range(n_down) for ll in range(n_right)]
        self.compute_distance()


Coupling.__torus = __torus


def __ibmqx(self, index):
    """
    Creates one of the IBM Quantum Experience coupling maps based on the index provided

    Args:
        index (integer between 2 and 5): specify which of the QX chips should be returned

    """
    dat = {2: {"name": "ibmqx2", "qubits": 5, "coupling_map": {0: [1, 2], 1: [2], 3: [2, 4], 4: [2]},
               "description": "IBM QX Sparrow: https://ibm.biz/qiskit-ibmqx2",
               "position": [[-1, 1], [1, 1], [0, 0], [1, -1], [-1, -1]]},
           3: {"name": "ibmqx3", "qubits": 16,
               "coupling_map": {0: [1], 1: [2], 2: [3], 3: [14], 4: [3, 5], 6: [7, 11], 7: [10], 8: [7], 9: [8, 10],
                                11: [10], 12: [5, 11, 13], 13: [4, 14], 15: [0, 14]},
               "description": "IBM QX Albatross: https://ibm.biz/qiskit-ibmqx3",
               "position": [[0, 0]] + [[xx, 1] for xx in range(8)] + [[7 - xx, 0] for xx in range(7)]},
           4: {"name": "ibmqx4", "qubits": 5, "coupling_map": {1: [0], 2: [0, 1, 4], 3: [2, 4]},
               "description": "IBM QX Raven: https://ibm.biz/qiskit-ibmqx4",
               "position": [[-1, 1], [1, 1], [0, 0], [1, -1], [-1, -1]]},
           5: {"name": "ibmqx5", "qubits": 16,
               "coupling_map": {1: [0, 2], 2: [3], 3: [4, 14], 5: [4], 6: [5, 7, 11], 7: [10], 8: [7], 9: [8, 10],
                                11: [10], 12: [5, 11, 13], 13: [4, 14], 15: [0, 2, 14]},
               "description": "IBM QX Albatross: https://ibm.biz/qiskit-ibmqx5",
               "position": [[0, 0]] + [[xx, 1] for xx in range(8)] + [[7 - xx, 0] for xx in range(7)]}
           }
    self.__clear()
    if index > 1 and index < 6:
        cl = dat.get(index)
        cdict = cl["coupling_map"]
        for v0, alist in cdict.items():
            for v1 in alist:
                regname = "q"
                self.add_edge((regname, v0), (regname, v1))
        self.name = cl["name"]
        self.description = cl["description"]
        self.pos = cl["position"]
        self.compute_distance()


Coupling.__ibmqx = __ibmqx


def __ibmq(self, index):
    """
    Creates one of the IBM Q coupling maps based on the index provided

    Args:
        index : specify which of the IBM Q chips should be returned

    """
    dat = {1: {"name": "QS1_1", "qubits": 20,
               "coupling_map": {0: [1, 5], 1: [0, 2, 6, 7], 2: [1, 7], 3: [4, 9], 5: [0, 6, 10, 11],
                                6: [1, 5, 7, 10, 11], 7: [1, 2, 6, 12, 13], 8: [4, 9, 12, 13], 9: [3, 4, 8, 14],
                                10: [5, 6, 11, 15], 11: [5, 6, 10, 12, 16, 17], 12: [7, 8, 11, 13, 16, 17],
                                13: [7, 8, 12, 14, 18, 19], 14: [9, 13, 18, 19], 15: [10, 16], 16: [11, 12, 15, 17],
                                17: [11, 12, 16, 18], 18: [13, 14, 17, 19], 19: [13, 14, 18]},
               "description": "IBM Q commercial: https://quantumexperience.ng.bluemix.net/qx/devices",
               "position": [[yy, 4 - xx] for xx in range(5) for yy in range(5)]}
           }
    if index > 0 and index < 2:
        cl = dat.get(index)
        cdict = cl["coupling_map"]
        for v0, alist in cdict.items():
            for v1 in alist:
                regname = "q"
                self.add_edge((regname, v0), (regname, v1))
        self.name = cl["name"]
        self.description = cl["description"]
        self.pos = cl["position"]
        self.compute_distance()


Coupling.__ibmq = __ibmq


def __add_defects(cmap, defects=0, unidir=False):
    # determine number of qubits in coupling map
    n = len(list(set([elem for nn in list(cmap.values())
                      for elem in nn] + list(cmap.keys()))))
    if defects > 0:
        cmap = __add_links(cmap, n=n, nl=round(defects), unidir=unidir)
    if defects < 0:
        cmap = __sub_links(cmap, n=n, nl=round(abs(defects)))
    return cmap


def __fexport(self, name=None):
    """
    Saves the coupling map as a json file to a subfolder named layouts

    Args:
        name : if specified the filename is used to save the coupling map
                   otherwise the name of the map is used
    """

    import json
    from os import path, getcwd
    if name is None:
        filename = self.name
        foldername = getcwd()
    else:
        (foldername, filename) = path.split(name)
    if len(foldername) == 0:
        foldername = getcwd()
    if len(filename) == 0:
        filename = self.name
    if '.' not in filename:
        filename += ".json"

    coupling_layout = {}
    coupling_layout["name"] = self.name
    coupling_layout["position"] = self.pos
    coupling_layout["coupling_map"] = self.get_dict()
    coupling_layout["description"] = self.description
    coupling_layout["qubits"] = len(self.qubits)
    with open(path.join(foldername, filename), 'w') as outfile:
        json.dump(coupling_layout, outfile)


Coupling.fexport = __fexport


def __fimport(self, name):
    """
    Loads a coupling layout that is given as a json file from disc

    Args:
        name (string): name of the coupling map that was used when saving
                       if no path is given an attempt is made to load from the current working directory
    """
    import json
    from os import path, getcwd
    (foldername, filename) = path.split(name)
    if len(foldername) == 0:
        foldername = getcwd()
    if '.' not in filename:
        filename += ".json"

    with open(path.join(foldername, filename), 'r') as infile:
        temp = json.load(infile)
        temp["coupling_map"] = {
            int(ii): kk for ii,
            kk in temp["coupling_map"].items()}
        self.__clear()
        self.name = temp["name"]
        self.pos = temp["position"]
        self.description = temp["description"]
        cdict = temp["coupling_map"]
        for v0, alist in cdict.items():
            for v1 in alist:
                regname = "q"
                self.add_edge((regname, v0), (regname, v1))
        self.compute_distance()


Coupling.fimport = __fimport


def __add_links(cmap, n, nl=0, unidir=False):
    # Add nl links to an existing coupling map
    # If it already exists then choose the next non existing link until full connectivity is reached.
    # TODO check if added links already exist in other direction and disallow
    # if unidir = True.
    for kk in range(nl):
        # create a random link
        ii = round(np.random.rand() * n - 0.5) % n  # choose a random source
        jj = round(np.random.rand() * n - 0.5) % n  # choose a random target
        # if it is a link onto itself shift the target by 1
        if ii == jj:
            jj = (jj + 1) % n
        # check if the source node exists in the coupling map
        if ii in cmap:
            # source node is in coupling map
            # store initial node indices
            ii_orig = (ii - 1) % n
            jj_orig = jj
            # search the target nodes until we find one that is not linked
            while jj in cmap[ii]:
                # the node is already in the list so increase the node number
                # by 1
                jj = (jj + 1) % n
                if jj == jj_orig:
                    # if the target node is again the original node number,
                    # increase the source node
                    ii = (ii + 1) % n
                    if ii not in cmap:
                        # if the source node is new add  the link
                        # but check first if it points to itself
                        if ii == jj:
                            jj = (jj + 1) % n
                        cmap[ii] = [jj]
                        jj = -jj  # prepare to exit from the while loop
                    elif ii == ii_orig or n * (n - 1) == sum([len(jj) for jj in coupling_map.values()]):
                        # if the increase in the source qubit index has brought us back to where we started
                        # we can assume that all-to-all connectivity was
                        # reached
                        return cmap  # no more links left to make
                    elif ii == jj:
                        jj = (jj + 1) % n
                        jj_orig = jj
            if jj >= 0:
                cmap[ii] = cmap[ii] + [jj]
        else:
            # source node is not yet in coupling map so just add the node
            cmap[ii] = [jj]
    return cmap


def __sub_links(cmap, n, nl=0):
    # Remove nl links from an existing coupling map until removal of more links
    # would lead to disjoint sets of qubits
    from copy import deepcopy
    for kk in range(nl):
        retry = n
        while retry > 0:
            retry = retry - 1
            cmap_copy = deepcopy(cmap)
            if len(cmap_copy) > 0:
                # TODO check interval boundaries for rand() function to avoid
                # mod
                ii = list(
                    cmap_copy.keys())[
                    round(
                        np.random.rand() *
                        len(cmap_copy) -
                        0.5) %
                    len(cmap_copy)]
                if len(cmap_copy[ii]) > 1:
                    jj = round(np.random.rand() *
                               len(cmap_copy[ii]) - 0.5) % len(cmap_copy[ii])
                    del (cmap_copy[ii][jj])
                else:
                    del (cmap_copy[ii])
            if not __is_disjoint(cmap_copy, n):
                cmap = cmap_copy
                retry = -10
        if retry == 0:
            # coupling_layout["description"] = coupling_layout["description"]+" Removed "+str(kk)+" links from lattice."
            return cmap
    # coupling_layout["description"] = coupling_layout["description"]+" Removed "+str(nl)+" links from lattice."
    return cmap


def __is_disjoint(coupling_map, nmax):
    # check if all nodes in the map are connected
    # TODO this should be replaced with the networkx function is_weakly_connected(G)
    # first check if all nodes are present in the coupling map
    if len(set([ii for jj in [list(coupling_map.keys())] +
                list(coupling_map.values()) for ii in jj])) < nmax:
        return True
    else:
        f_map = np.zeros(nmax)  # empty filling map
        f_val = 1
        f_ind = 0  # start flooding from node 0
        f_map[f_ind] = f_val
        while min(f_map) == 0 and f_val < nmax:
            # determine all links from node f_ind not taking directionality into account
            # it includes the backwards link but not the node itself
            f_links = [jj if ii == f_ind else [ii] for (
                ii, jj) in coupling_map.items() if ii == f_ind or f_ind in jj]
            f_links = [jj for ii in f_links for jj in ii]
            # choose smallest filling value node
            f_min = nmax  # initialize with a large value
            for ii in f_links:  # find linked node with smallest flooding level if multiple choose first one
                f_val = f_map[ii]
                if f_val < f_min:
                    f_min = f_val
                    f_ind = ii
            # increase flooding level at index f_ind
            f_map[f_ind] = f_map[f_ind] + 1
        # return true if there are unflooded nodes in flooding map
        return (min(f_map) == 0)


# Override old __init__ to accept a "pos" keyword in addition to all previous ones
# if the generate method already exist don't do anything since this may
# lead to a recursive definition otherwise

if getattr(Coupling, "gen_map", None) is None:
    oldinit = Coupling.__init__

    def __newinit(self, couplingdict=None, **kwargs):
        if "layout" in kwargs.keys():
            self.__clear()
            self.gen_map(**kwargs)
        else:
            if "pos" in kwargs.keys():
                self.pos = kwargs["pos"]
                kwargs.pop("pos")
            else:
                self.pos = None

            if "name" in kwargs.keys():
                self.name = kwargs["name"]
                kwargs.pop("name")
            else:
                self.name = "cmap" + time.strftime("%y%m%d%H%M%S")

            if "description" in kwargs.keys():
                self.description = kwargs["description"]
                kwargs.pop("description")
            else:
                self.description = "No description of coupling map given."
            oldinit(self, couplingdict=couplingdict)

    Coupling.__init__ = __newinit


def __gen_map(self, layout=None, **kwargs):
    if layout is not None:
        if layout == "linear":
            self.__linear(**kwargs)
        elif layout == "circle":
            self.__circle(**kwargs)
        elif layout == "rect":
            self.__rect(**kwargs)
        elif len(layout) > 5 and layout[:5] == "ibmqx":
            self.__ibmqx(int(layout[5:]))
        elif len(layout) > 4 and layout[:4] == "ibmq":
            self.__ibmq(int(layout[4:]))
        elif layout == "torus":
            self.__torus(**kwargs)
        else:
            raise ValueError(
                'The layout "' +
                str(layout) +
                '" is not a known layout!')
    else:
        raise ValueError('Valid layout name required!')


Coupling.gen_map = __gen_map
