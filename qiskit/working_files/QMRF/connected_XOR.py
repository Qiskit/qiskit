import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize,linewidth=1024)

import itertools

I = np.array([[1,0],[0,1]])
Z = np.array([[1,0],[0,-1]])

C = [[0,1,2],[3,4,5],[2,3]] # clique structure
n = 6 # number of (qu)bits
X = list(itertools.product([0, 1], repeat=n)) # state space of size 2^n

# Mark all elements in the diagonal whose binary representation has the bits in index_set asserted	
def Phi_slow(c,y):
	result = np.zeros(2**n * 2**n).reshape(2**n, 2**n)
	# This generates all ones in Phi with complexity 2^n!! We need something else here!
	for j,x in enumerate(X):
		valid = True
		for i,v in enumerate(c):
			if y[i] != x[v]:
				valid = False
				break
		if valid:
			result[j,j] = 1
			
	return result
		
# Mark all elements in the diagonal whose binary representation has the bits in index_set asserted	
def Phi_fast(c,y):
	result = 1
	
	plus  = [v for i,v in enumerate(c) if not y[i]]
	minus = [v for i,v in enumerate(c) if     y[i]]
	
	s = 2.0**(len(plus) + len(minus))
	
	# This is the solution
	for i in range(n):
		f = I
		if i in minus:
			f = I-Z
		elif i in plus:
			f = I+Z

		result = np.kron(result,f)
		
	return result / s
	
def isXOR(x,y,z):
	r = (z == (x!=y))
	# print(x,y,z,r) # yes, this checks indeed wheter z = x XOR y
	return r


def gen_Hamiltonian(mode):
	H = np.zeros(2**n * 2**n).reshape(2**n, 2**n) # Hamitonian of size 2^n X 2^n

	for l,c in enumerate(C):
		Y = list(itertools.product([0, 1], repeat=len(c)))
		for y in Y:
			if mode == 'fast':
				Phi = Phi_fast(c,y)
			else:
				Phi = Phi_slow(c,y)			

			theta = -1 # theta will be negated when 3-cliques are not in valid XOR state or the 2-clique is 00 and 11
			if (l==0 or l==1) and isXOR(y[0],y[1],y[2]):
				theta *= -1
			elif l==2 and (y[0] != y[1]):
				theta *= -1

			H += theta * Phi
			
	H[H>0] =  3
	H[H<0] = -3
	
	return H

H0 = gen_Hamiltonian('slow')
H1 = gen_Hamiltonian('fast')

print(hex(hash(H0.tobytes())))
print(hex(hash(H1.tobytes())))

assert (H0 == H1).all()

print(H0)
print(H1.tolist())
