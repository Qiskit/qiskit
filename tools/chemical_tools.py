""" 
Chemical_tools 

Set of functions used to map fermionic Hamiltonians to qubits 

Author: Antonio Mezzacapo 

"""

from tools.pauli import Pauli, label_to_pauli
import numpy as np



def parity_set(j,dim):
    
    indexes=[]
    
    if dim%2==0:
        
        if j<dim/2:
            
            indexes=np.append(indexes,parity_set(j,dim/2))
        
        else:
            
            indexes=np.append(np.append(parity_set(j-dim/2,dim/2)+dim/2,dim/2-1))
    
    return indexes 

def update_set(j,dim):
    
    indexes=[]
    
    if dim%2==0:
        
        if j<dim/2:
            
            indexes=np.append(np.append(dim-1,update_set(j,dim/2)))
            
        else:
            
            indexes=np.append(indexes,update_set(j-dim/2,dim/2)+dim/2)
                              
    return indexes  

def filp_set(j,dim):
    
    if dim%2==0:
        
        if j<dim/2:
            
            indexes=np.append(indexes,flip_set(j,dim/2))
            
        elif j>=dim/2 and j<dim-1:
            
            indexes=np.append(indexes,flip_set(j-dim/2,dim/2)+dim/2)
            
        else:
            
            indexes=np.append(np.append(indexes,flip_set(j-dim/2,dim/2)+dim/2),dim/2-1)
            
    return indexes 
            
            
                

def fermionic_maps(h1,h2,map_type,out_file):

    pauli_list=[]
    
    #n=len(h1) # number of fermionic modes / qubits  
    
    n=5
    
        #####   DEFINING MAPPED FERMIONIC OPERATORS    #####
    a=[]
    
    if map_type=='JORDAN_WIGNER':
        
        for i in range(n):
            
           
            Xv=np.append(np.append(np.zeros((1,i-1)),0),np.ones((1,n-i))) 
            Xw=np.append(np.append(np.zeros((1,i-1)),1),np.zeros((1,n-i)))
            Yv=np.append(np.append(np.zeros((1,i-1)),1),np.ones((1,n-i)))
            Yw=np.append(np.append(np.zeros((1,i-1)),1),np.zeros((1,n-i)))
            
    if map_type=='PARITY':
        
        for i in range(n):
            
            if i>1:
           
                Xv=np.append(np.append(np.zeros((1,i-1)),(1,0)),np.zeros((1,n-i-1)))         
                Xw=np.append(np.append(np.zeros((1,i-1)),[0,1]),np.ones((1,n-i-1)))         
                Yv=np.append(np.append(np.zeros((1,i-1)),[0,1]),np.zeros((1,n-i-1)))
                Yw=np.append(np.append(np.zeros((1,i-1)),[0,1]),np.ones((1,n-i-1)))
            
            elif i>0:
                
                Xv=np.append((1,0),np.zeros((1,n-i-1)))         
                Xw=np.append([0,1],np.ones((1,n-i-1)))         
                Yv=np.append([0,1],np.zeros((1,n-i-1)))
                Yw=np.append([0,1],np.ones((1,n-i-1)))    
            
            else:
                
                Xv=np.append(0,np.zeros((1,n-i-1)))         
                Xw=np.append(1,np.ones((1,n-i-1)))         
                Yv=np.append(1,np.zeros((1,n-i-1)))
                Yw=np.append(1,np.ones((1,n-i-1)))   
                
    if map_type='BINARY_TREE':
        
            

    a.append((Pauli(Xv,Xw),Pauli(Yv,Yw))) # defines the two Pauli components of a
    
    
    
    for i in range(n):
        print(a[i][0].to_label())
        print(a[i][1].to_label())
                                                                
            
        
        
    return 0    
    
    