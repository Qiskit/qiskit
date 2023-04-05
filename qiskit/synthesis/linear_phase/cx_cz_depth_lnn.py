# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2022
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Given -CZ-CX- transformation (a layer consisting only CNOT gates 
    followed by a layer consisting only CZ gates)
Return a depth-5n circuit implementation of the -CX-CZ- transformation over LNN.

Input:
    Mx: n*n invertable binary matrix representing a -CX- transformation
    Mz: n*n symmetric binary matrix representing a -CZ- circuit
    
Output:
    qc: QuantumCircuit object containing a depth-5n circuit to implement -CZ-CX-

References:
    [1] S. A. Kutin, D. P. Moulton, and L. M. Smithline, "Computation at a distance," 2007.
    [2] D. Maslove and W. Yang, "CNOT circuits need little help to implement arbitrary 
        Hadamard-free Clifford transformations they generate," 2022.
"""

import numpy as np
from copy import deepcopy

from qiskit.exceptions import QiskitError
from qiskit.circuit import QuantumCircuit
from qiskit.synthesis.linear.linear_matrix_utils import (calc_inverse_matrix)

from qiskit.synthesis.linear.linear_depth_lnn import (
    _optimize_cx_circ_depth_5n_line
    )
        
def _initializeS(Mz):
    '''
    Given a CZ layer (represented as an n*n CZ matrix Mz)
    Return a scheudle of phase gates implementing Mz in a SWAP-only netwrok 
    (Ref. [Alg 1, 2])
    '''
    n = len(Mz)
    S = np.zeros((n,n),dtype = int)
    for i, j in zip(*np.where(Mz)):
        if i>j:
            continue
            
        S[i,j] = 3
        S[i,i] += 1
        S[j,j] += 1
        
    return S  

def _shuffle(l,odd):
    '''
    Given a list of indices l and boolean odd indicating odd/even layers,
    Shuffle the indices in l by swapping adjacent elements
    (Ref. [Fig.2, 2])
    '''
    swapped = [v for p in zip(l[1::2],l[::2]) for v in p] 
    return swapped + l[-1:] if odd else swapped
    
def _makeSeq(n):
    '''
    Given the width of the circuit n, 
    Return the label of the boxes in order from left to right, top to bottom
    (Ref. [Fig.2, 2])
    '''
    seq = []
    l = list(range(n-1,-1,-1))

    for i in range(n):
        r = _shuffle(l,n%2) if i%2==0 else l[0:1] + _shuffle(l[1:],(n+1)%2)
        seq += list([(min(i),max(i)) for i in zip(l[::2],r[::2]) if i[0]!=i[1]])
        l = r
    return seq

def _swapPlus(instructions,seq):
    '''
    Given CX instructions (Ref. [Thm 7.1, 1]) and the labels of all boxes,
    Return a list of labels of the boxes that is SWAP+ in descending order 
        * Assumes the instruction gives gates in the order from top to bottom, 
          from left to right
    '''
    instr = deepcopy(instructions)
    swapPs = set()
    for i,j in reversed(seq):
        CNOT1 = instr.pop()
        CNOT2 = instr.pop()

        if instr == [] or instr[-1]!= CNOT1:
            #Only two CNOTs on same set of controls -> this box is SWAP+
            swapPs.add((i,j)) 
        else:
            CNOT3 = instr.pop()
    return swapPs

def _updateS(n, S, swapPs):
    '''
    Given S initialized to induce a CZ circuit in SWAP-only network and list of SWAP+ boxes
    Update S for each SWAP+ according to Algorithm 2 [2]
    '''
    l = list(range(n))
    Pn = l[-3::-2]+l[-2::-2][::-1]
    orderComp = np.argsort(Pn[::-1])

    # Go through each box by descending layer order
    
    for i in Pn:
        for j in range(i+1,n):
            if (i,j) not in swapPs:
                continue
            # we need to correct for the effected linear functions:

            # We first correct type 1 and type 2 by switching 
            # the phase applied to c_j and c_i+c_j
            S[j,j] , S[i,j] = S[i,j], S[j,j]

            # Then, we go through all the boxes that permutes j BEFORE box(i,j) and update:

            for k in range(n): #all boxes that permutes j
                if i==k or j==k:
                    continue
                if orderComp[min(k,j)] < orderComp[i] and S[min(k,j),max(k,j)]%4!=0: 
                    phase = S[min(k,j),max(k,j)]
                    S[min(k,j),max(k,j)] = 0

                    #Step 1, apply phase to c_i, c_j, c_k
                    for l in [i,j,k]:
                        S[l,l] = (S[l,l]+phase*3)%4

                    #Step 2, apply phase to c_i+ c_j, c_i+c_k, c_j+c_k:
                    for l1,l2 in [(i,j), (i,k),(j,k)]:
                        ls = min(l1,l2)
                        lb = max(l1,l2)
                        S[ls,lb] = (S[ls,lb]+phase*3)%4
    return S
                    
def _apply_S_to_NW_circuit(n,S,seq, swapPs):
    '''
    Given 
        Width of the circuit (int n)
        A CZ circuit, represented by the n*n phase schedule S
        A CX circuit, represented by box-labels (seq) and whether the box is SWAP+ (swapPs)
            *   This circuit corresponds to the CX tranformation that tranforms a matrix to
                a NW matrix (Ref. [Prop.7.4, 1])
    Return a QuantumCircuit that computes S and CX
    '''
    cir = QuantumCircuit(n)
    
    wires = list(zip(range(n),range(1,n)))
    wires = wires[::2]+wires[1::2]  
  
    
    for (i,(j,k)) in zip(range(len(seq)-1,-1,-1),reversed(seq)):
        
        w1, w2 = wires[i%(n-1)]

        p = S[j,k]
        
        if (j,k) not in swapPs:
            cir.cnot(w1,w2)
        
        cir.cnot(w2,w1)
        
        if p%4 == 0:
            pass
        elif p%4 == 1:
            cir.sdg(w2)
        elif p%4 == 2:
            cir.z(w2)
        elif p%4 == 3:
            cir.s(w2)
        
        cir.cnot(w1,w2)
        
    for i in range(n):
        p = S[n-1-i,n-1-i]
        if p%4 == 0:
            continue
        elif p%4 == 1:
            cir.sdg(i)
        elif p%4 == 2:
            cir.z(i)
        elif p%4 == 3:
            cir.s(i)  
            
    return cir



def synth_cx_cz_line_my(Mx,Mz):
    '''
    Given 
        -CX- circuit, represented by n*n binary, invertible matrix Mx
        -CZ- circuit, repredented by n*n binary, symmetric  matrix Mz
            where n is the width of the circuit
    Return a QuantumCircuit object, qc, that implements the -CZ-CX- in two-qubit depth at most 5n
    '''

    # Find circuits implementing Mx by Proposition 7.3 and Proposition 7.4 of [1]

    n = len(Mx)
    Mx = calc_inverse_matrix(Mx)

    cx_instructions_rows_m2nw, cx_instructions_rows_nw2id = _optimize_cx_circ_depth_5n_line(Mx)
    
    #Meanwhile, also build the -CZ- circuit via Phase gate insertions as per Algorithm 2 [2]
    S = _initializeS(Mz)  
    seq = _makeSeq(n)
    swapPs = _swapPlus(cx_instructions_rows_nw2id, seq)

    _updateS(n, S, swapPs)
    
    qc = _apply_S_to_NW_circuit(n,S,seq, swapPs)
    
    for i,j in reversed(cx_instructions_rows_m2nw):
        qc.cx(i,j)
    
    return qc
