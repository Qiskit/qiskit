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
File: fermion_to_qubit_tools.py

Description:

A set of functions that map fermionic Hamiltonians to qubit Hamiltonians. fermionic_maps maps a one- and two-electron
fermionic operators into a qubit Hamiltonian, according to three types of mappings.

References:
- E. Wigner and P. Jordan., Über das Paulische Äguivalenzverbot, Z. Phys., 47:631 (1928).
- S. Bravyi and A. Kitaev. Fermionic quantum computation, Ann. of Phys., 298(1):210–226 (2002).
- A. Tranter, S. Sofia, J. Seeley, M. Kaicher, J. McClean, R. Babbush, P. Coveney, F. Mintert, F. Wilhelm, and P. Love. The Bravyi–Kitaev transformation: Properties and applications. Int. Journal of Quantum Chemistry, 115(19):1431–1441 (2015).
- S. Bravyi, J. M. Gambetta, A. Mezzacapo, and K. Temme, arXiv e-print arXiv:1701.08213 (2017).


"""

from tools.qi.pauli import Pauli, label_to_pauli,sgn_prod
import numpy as np
from tools.apps.optimization import Hamiltonian_from_file


"""
The three functions parity_set, update_set and flip_set define three sets of qubit indices, associated to each fermionic mode j in a set of n qubits, used in the binary-tree mapping.

"""

def parity_set(j,n):

    indexes=np.array([])
    if n%2==0:
        if j<n/2:
            indexes=np.append(indexes,parity_set(j,n/2))
        else:
            indexes=np.append(indexes,np.append(parity_set(j-n/2,n/2)+n/2,n/2-1))
    return indexes

def update_set(j,n):

    indexes=np.array([])
    if n%2==0:
        if j<n/2:
            indexes=np.append(indexes,np.append(n-1,update_set(j,n/2)))
        else:
            indexes=np.append(indexes,update_set(j-n/2,n/2)+n/2)
    return indexes

def flip_set(j,n):

    indexes=np.array([])
    if n%2==0:
        if j<n/2:
            indexes=np.append(indexes,flip_set(j,n/2))
        elif j>=n/2 and j<n-1:
            indexes=np.append(indexes,flip_set(j-n/2,n/2)+n/2)
        else:
            indexes=np.append(np.append(indexes,flip_set(j-n/2,n/2)+n/2),n/2-1)
    return indexes

def pauli_term_append(pauli_term,pauli_list,threshold):

    """
    The function appends pauli_term to pauli_list if is not present in pauli_list.
    If present in the list adjusts the coefficient of the existing pauli. If the new
    coefficient is less than threshold the pauli term is deleted from the list
    """

    found=False

    if np.absolute(pauli_term[0])>threshold:

        if (not not pauli_list):   # if the list is not empty

            for i in range(len(pauli_list)):

                if pauli_list[i][1].to_label()==pauli_term[1].to_label():   # check if the new pauli belongs to the list

                    pauli_list[i][0]+=pauli_term[0]    # if found renormalize the coefficient of existent pauli

                    if np.absolute(pauli_list[i][0])<threshold: # remove the element if coeff. value is now less than threshold
                        del pauli_list[i]

                    found=True
                    break

            if found==False:       # if not found add the new pauli
                pauli_list.append(pauli_term)

        else:
            pauli_list.append(pauli_term)      # if list is empty add the new pauli



    return pauli_list




def fermionic_maps(h1,h2,map_type,out_file=None,threshold=0.000000000001):


    """ Takes fermionic one and two-body operators in the form of numpy arrays with real entries, e.g.
        h1=np.zeros((n,n))
        h2=np.zeros((n,n,n,n))
        where n is the number of fermionic modes, and gives a pauli_list of mapped pauli terms and
        coefficients, according to the map_type specified, with values

        map_type:
        JORDAN_WIGNER
        PARITY
        BINARY_TREE

        the notation for the two-body operator is the chemists' one,
        h2(i,j,k,m) a^dag_i a^dag_k a_m a_j

        Options:
        - writes the mapped pauli_list to a file named out_file given as an input (does not do this as default)
        - neglects mapped terms below a threshold defined by the user (default is 10^-12)

    """

    pauli_list=[]

    n=len(h1) # number of fermionic modes / qubits

    """
    ####################################################################
    ############   DEFINING MAPPED FERMIONIC OPERATORS    ##############
    ####################################################################
    """

    a=[]

    if map_type=='JORDAN_WIGNER':

        for i in range(n):


            Xv=np.append(np.append(np.ones(i),0),np.zeros(n-i-1))
            Xw=np.append(np.append(np.zeros(i),1),np.zeros(n-i-1))
            Yv=np.append(np.append(np.ones(i),1),np.zeros(n-i-1))
            Yw=np.append(np.append(np.zeros(i),1),np.zeros(n-i-1))

            # defines the two mapped Pauli components of a_i and a_i^\dag, according to a_i -> (a[i][0]+i*a[i][1])/2, a_i^\dag -> (a_[i][0]-i*a[i][1])/2
            a.append((Pauli(Xv,Xw),Pauli(Yv,Yw)))


    if map_type=='PARITY':

        for i in range(n):

            if i>1:

                Xv=np.append(np.append(np.zeros(i-1),[1,0]),np.zeros(n-i-1))
                Xw=np.append(np.append(np.zeros(i-1),[0,1]),np.ones(n-i-1))
                Yv=np.append(np.append(np.zeros(i-1),[0,1]),np.zeros(n-i-1))
                Yw=np.append(np.append(np.zeros(i-1),[0,1]),np.ones(n-i-1))

            elif i>0:

                Xv=np.append((1,0),np.zeros(n-i-1))
                Xw=np.append([0,1],np.ones(n-i-1))
                Yv=np.append([0,1],np.zeros(n-i-1))
                Yw=np.append([0,1],np.ones(n-i-1))

            else:

                Xv=np.append(0,np.zeros(n-i-1))
                Xw=np.append(1,np.ones(n-i-1))
                Yv=np.append(1,np.zeros(n-i-1))
                Yw=np.append(1,np.ones(n-i-1))

            # defines the two mapped Pauli components of a_i and a_i^\dag, according to a_i -> (a[i][0]+i*a[i][1])/2, a_i^\dag -> (a_[i][0]-i*a[i][1])/2
            a.append((Pauli(Xv,Xw),Pauli(Yv,Yw)))


    if map_type=='BINARY_TREE':


        # FIND BINARY SUPERSET SIZE

        bin_sup=1
        while n>np.power(2,bin_sup):
            bin_sup+=1

        # DEFINE INDEX SETS FOR EVERY FERMIONIC MODE

        update_sets=[]
        update_pauli=[]

        parity_sets=[]
        parity_pauli=[]

        flip_sets=[]
        flip_pauli=[]

        remainder_sets=[]
        remainder_pauli=[]


        for j in range(n):

            update_sets.append(update_set(j,np.power(2,bin_sup)))
            update_sets[j]=update_sets[j][update_sets[j]<n]

            parity_sets.append(parity_set(j,np.power(2,bin_sup)))
            parity_sets[j]=parity_sets[j][parity_sets[j]<n]

            flip_sets.append(flip_set(j,np.power(2,bin_sup)))
            flip_sets[j]=flip_sets[j][flip_sets[j]<n]

            remainder_sets.append(np.setdiff1d(parity_sets[j],flip_sets[j]))




            update_pauli.append(Pauli(np.zeros(n),np.zeros(n)))
            parity_pauli.append(Pauli(np.zeros(n),np.zeros(n)))
            remainder_pauli.append(Pauli(np.zeros(n),np.zeros(n)))

            for k in range(n):

                if np.in1d(k,update_sets[j]):

                    update_pauli[j].w[k]=1

                if np.in1d(k,parity_sets[j]):

                    parity_pauli[j].v[k]=1

                if np.in1d(k,remainder_sets[j]):

                    remainder_pauli[j].v[k]=1

            Xj=Pauli(np.zeros(n),np.zeros(n))
            Xj.w[j]=1
            Yj=Pauli(np.zeros(n),np.zeros(n))
            Yj.v[j]=1
            Yj.w[j]=1

            # defines the two mapped Pauli components of a_i and a_i^\dag, according to a_i -> (a[i][0]+i*a[i][1])/2, a_i^\dag -> (a_[i][0]-i*a[i][1])/2
            a.append((update_pauli[j]*Xj*parity_pauli[j],update_pauli[j]*Yj*remainder_pauli[j]))


    """
    ####################################################################
    ############    BUILDING THE MAPPED HAMILTONIAN     ################
    ####################################################################
    """


    """
    #######################    One-body    #############################
    """

    for i in range(n):
        for j in range(n):
            if h1[i,j]!=0:
                for alpha in range(2):
                    for beta in range(2):

                            pauli_prod=sgn_prod(a[i][alpha],a[j][beta])
                            pauli_term=[  h1[i,j]*1/4*pauli_prod[1]*np.power(-1j,alpha)*np.power(1j,beta),  pauli_prod[0]  ]
                            pauli_list=pauli_term_append(pauli_term,pauli_list,threshold)



    """
    #######################    Two-body    #############################
    """

    for i in range(n):
        for j in range(n):
            for k in range(n):
                for m in range(n):

                    if h2[i,j,k,m]!=0:

                        for alpha in range(2):
                            for beta in range(2):
                                for gamma in range(2):
                                    for delta in range(2):

                                        """
                                        # Note: chemists' notation for the labeling, h2(i,j,k,m) adag_i adag_k a_m a_j
                                        """

                                        pauli_prod_1=sgn_prod(a[i][alpha],a[k][beta])
                                        pauli_prod_2=sgn_prod(pauli_prod_1[0],a[m][gamma])
                                        pauli_prod_3=sgn_prod(pauli_prod_2[0],a[j][delta])

                                        phase1=pauli_prod_1[1]*pauli_prod_2[1]*pauli_prod_3[1]
                                        phase2=np.power(-1j,alpha+beta)*np.power(1j,gamma+delta)

                                        pauli_term=[h2[i,j,k,m]*1/16*phase1*phase2,pauli_prod_3[0]]

                                        pauli_list=pauli_term_append(pauli_term,pauli_list,threshold)


    """
    ####################################################################
    #################          WRITE TO FILE         ###################
    ####################################################################
    """


    if out_file!= None:
        out_stream=open(out_file,'w')

        for pauli_term in pauli_list:
            out_stream.write(pauli_term[1].to_label()+'\n')
            out_stream.write('%.15f' % pauli_term[0].real+'\n')

        out_stream.close()

    return pauli_list



def two_qubit_reduction(ham_in,m,out_file=None,threshold=0.000000000001):

    """
    This function takes in a mapped fermionic Hamiltonian with an even number of modes n, obtained with the parity (for every even n) or binary-tree mapping (in case the number of modes is a power of 2, n=2^k, k integer) and removes two qubits at positions n/2,n according to the total number of particles m.


    ham_in can be both a pauli_list type or a string with a input Hamiltonian text filename.
    The function returns a pauli_list and optionally creates a Hamiltonian text file with name out_file
    m is the number of particles (e.g. electrons) in the system

    """

    ham_out=[]


    if m%4==0:
        par_1=1
        par_2=1
    elif m%4==1:
        par_1=-1
        par_2=-1    # could be also +1, +1/-1 are degenerate spin-parity sectors
    elif m%4==2:
        par_1=1
        par_2=-1
    else:
        par_1=-1
        par_2=-1    # could be also +1, +1/-1 are degenerate spin-parity sectors


    if type(ham_in) is str:

        file_name=ham_in
        ham_in=Hamiltonian_from_file(ham_in)    # conversion from Hamiltonian text file to pauli_list

    # number of qubits
    n=len(ham_in[0][1].v)

    for pauli_term in ham_in:#loop over Pauli terms

        coeff_out=pauli_term[0]

        if pauli_term[1].v[n//2-1]==1 and pauli_term[1].w[n//2-1]==0:  # Z operator encountered at qubit n/2-1

            coeff_out=par_2*coeff_out

        if pauli_term[1].v[n-1]==1 and pauli_term[1].w[n-1]==0:  # Z operator encountered at qubit n-1

            coeff_out=par_1*coeff_out

        v_temp=[]
        w_temp=[]
        for j in range(n):
            if j!=n//2-1 and j!=n-1:

                v_temp.append(pauli_term[1].v[j])
                w_temp.append(pauli_term[1].w[j])


        pauli_term_out=[coeff_out,Pauli(v_temp,w_temp)]
        ham_out=pauli_term_append(pauli_term_out,ham_out,threshold)




    """
    ####################################################################
    #################          WRITE TO FILE         ###################
    ####################################################################
    """

    if out_file!=None:

            out_stream=open(out_file,'w')

            for pauli_term in ham_out:
                out_stream.write(pauli_term[1].to_label()+'\n')
                out_stream.write('%.15f' % pauli_term[0].real+'\n')

            out_stream.close()




    return ham_out
