""" 
chemical_tools 

Set of functions used to map fermionic Hamiltonians to qubits 

Author: Antonio Mezzacapo 

"""

from tools.pauli import Pauli, label_to_pauli,sgn_prod
import numpy as np



def parity_set(j,dim):
    
    indexes=np.array([])   
    if dim%2==0:
        if j<dim/2:
            indexes=np.append(indexes,parity_set(j,dim/2))
        else:
            indexes=np.append(indexes,np.append(parity_set(j-dim/2,dim/2)+dim/2,dim/2-1))
    return indexes 

def update_set(j,dim):
    
    indexes=np.array([])
    if dim%2==0:
        if j<dim/2:
            indexes=np.append(indexes,np.append(dim-1,update_set(j,dim/2)))
        else:
            indexes=np.append(indexes,update_set(j-dim/2,dim/2)+dim/2)                 
    return indexes  

def flip_set(j,dim):
    
    indexes=np.array([])
    if dim%2==0:
        if j<dim/2:
            indexes=np.append(indexes,flip_set(j,dim/2))
        elif j>=dim/2 and j<dim-1:
            indexes=np.append(indexes,flip_set(j-dim/2,dim/2)+dim/2)
        else:
            indexes=np.append(np.append(indexes,flip_set(j-dim/2,dim/2)+dim/2),dim/2-1)
    return indexes 

def pauli_sel_add(pauli_term,pauli_list,threshold):

    # appends pauli_term to pauli_list if is not present in pauli_list. 
    # If present adjusts the coefficient of the existing pauli 
 
    found=False
    
    if not not pauli_list:   # if the list is not empty
        
        for i in range(len(pauli_list)):   
            
            if pauli_list[i][1].to_label()==pauli_term[1].to_label():   # check if the new pauli belong to the list 
                
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

    pauli_list=[]
    
    n=len(h1) # number of fermionic modes / qubits  
    
    print('length h')
    print(n)
    
    
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
            
            a.append((Pauli(Xv,Xw),Pauli(Yv,Yw))) # defines the two Pauli components of a
          
            
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
                
            a.append((Pauli(Xv,Xw),Pauli(Yv,Yw))) # defines the two Pauli components of a
           
                
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
            
            a.append((update_pauli[j]*Xj*parity_pauli[j],update_pauli[j]*Yj*remainder_pauli[j]))
            
          
    """            
    ####################################################################
    ############    BUILDING THE MAPPED HAMILTONIAN     ################
    ####################################################################
    """
            
            
            
#    for i in range(n):
        
 #       print(a[i][0].to_label())
  #      print(a[i][1].to_label())
        
        
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
                            pauli_list=pauli_sel_add(pauli_term,pauli_list,threshold)        
            

 
    """
    #######################    Two-body    #############################
    """
    print('CHECK')
    print(h2[0,1,1,0])
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for m in range(n):
                    
                    print([i,j,k,m])
                    
                    if h2[i,j,k,m]!=0:
                        print('INSIDE')
                        for alpha in range(2):
                            for beta in range(2):
                                for gamma in range(2):
                                    for delta in range(2):
                                        
                                        """
                                        # Note: chemists' notation for the labeling, h2(i,j,k,m) adag_i adag_k a_m a_j
                                        """
                                        
                                        pauli_prod_1=sgn_prod(a[i][alpha],a[k][beta])
                                        
                                        print('ai')
                                        print(a[i][alpha].to_label())
                                        print('ak')
                                        print(a[k][beta].to_label())
                                        print('1 sign')
                                        print(pauli_prod_1[1])
                                       
                                        
                                        
                                        pauli_prod_2=sgn_prod(pauli_prod_1[0],a[m][gamma])
                                        
                                       
                                        
                                        
                                        print('am')
                                        print(a[m][gamma].to_label())
                                        print('2 sign')
                                        print(pauli_prod_2[1])
                                        
                                        
                                        
                                        pauli_prod_3=sgn_prod(pauli_prod_2[0],a[j][delta])
                                        
                                        
                                        print('aj')
                                        print(a[j][delta].to_label())
                                        print('pauli_prod_3')
                                        print(pauli_prod_3[0].to_label())
                                        print(pauli_prod_3[1])
                                        
                                        
                                        
                                        phase1=pauli_prod_1[1]*pauli_prod_2[1]*pauli_prod_3[1]
                                        print('phase1')
                                        print(phase1)
                                        
                                        phase2=np.power(-1j,alpha+beta)*np.power(1j,gamma+delta)
                                        print('phase2')
                                        print(phase2)
                                        print(pauli_prod_3[0].to_label())
                                        
                                        pauli_term=[h2[i,j,k,m]*1/8*phase1*phase2 ,  pauli_prod_3[0]  ]
                                        print(pauli_term[0])
                                        
                                        pauli_list=pauli_sel_add(pauli_term,pauli_list,threshold)
                                        
                        
                        
                       
                    
                    
                    
                    
                    
                    
                    
    print('Check final ham')
    for pauli_term in pauli_list:
        
        
        print(pauli_term[1].to_label())
        print(pauli_term[0])
    
    
    
  
       
                                                                
            
        
        
    return pauli_list 
    
    