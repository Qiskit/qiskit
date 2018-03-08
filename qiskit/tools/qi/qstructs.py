'''
This file contains data structures useful in Quantum Information.
Examples are: QuantumState, DensityMatrix, UnitaryOperator.
Also contains useful functions, for example randcomplex.
'''

import sys
import numbers
import math
import numpy as np


# ** phase **
def phase(angle):
    '''
    Returns e^(i*angle)
    '''
    return math.e**(np.complex(0,1)*angle)


# ** is_close **
def is_close(a, b, rel_tol = 1e-09, abs_tol = 1e-09):
    """
    Returns true is 'a' and 'b' are close to each other.
    'a' and 'b' can be numbers of any type (integer, float, complex)
    and arrays of numbers.

    We use the existing function math.is_close

    numpy has a function for arrays, numpy.is_close,
    but it is not working well,
    therefore we will not call it.
    https://stackoverflow.com/questions/48156460/isclose-function-in-numpy-is-different-from-math
    """
    
    if isinstance(a, numbers.Number):
        
        if isinstance(a, np.complex):
            c = a
        else:
            c = np.complex(a, 0)

        if isinstance(b, np.complex):
            d = b
        else:
            d = np.complex(b, 0)
        
        return (math.isclose(np.real(c), np.real(d), rel_tol = rel_tol, abs_tol = abs_tol) and
                math.isclose(np.imag(c), np.imag(d), rel_tol = rel_tol, abs_tol = abs_tol))
    else:
        return all([is_close(x, y, rel_tol, abs_tol) for (x,y) in zip(np.array(a), np.array(b))])


# ** is_close_to_id **
def is_close_to_id(mat):
    """
    Checks if a matrix of complex numbers is equal (up to numerical fluctuations) to the identity matrix
    """
    # FIXME: verify that mat is a square matrix
    return is_close(mat, np.identity(len(mat)))


# ** randcomplex **
def randcomplex(n):
    """ Create a random vector of complex numbers """
    
    assert float(n).is_integer() and n >= 1, \
           'A request to generate a complex array of length ' + str(n) + '.' \
           'Length must be an integer strictly greater than 0.'

    real_array = np.random.rand(n, 2)
    return np.array([complex(row[0], ((-1)**np.random.randint(2))*row[1]) for row in real_array ])


# ** arraydot **
def arraydot(a, b):
    """
    For two vectors of matrices a=[A_1,...,A_n] and b=[B_1,...,B_n],
    compute the sum of A_i*B_i over all i=1,...,n
    """
    return np.tensordot(a, b, axes=1)


# ** ProbabilityDistribution **
class ProbabilityDistribution:
    """A vector of real non-negative numbers whose sum is 1"""
    
    def __init__(self, probs = None, n = None, not_normalized = None):
        """
        If probs is specified then this is the probability vector used for initialization,
        otherwise we create a random probability vector of length n.
        By default, probs is required to have sum 1.
        There is an option to set not_normalized to false,
        in which case we normalize the probability vector.
        """

        assert (n is None) or (float(n).is_integer() and n >= 1), \
               'A request to generate a probability distribution of length ' + str(n) + '. ' \
               'Length must be an integer strictly greater than 0.'
        
        if probs is None:
            # Will generate a random probability vector with the given length

            assert not_normalized is None, \
                  "Constructor of ProbabilityDistribution: " \
                  "argument 'not_normalized' is irrelevant " \
                  "if argument 'probs' is None"

            not_normalized = True

            assert n is not None, \
                   "Constructor of ProbabilityDistribution expects either " \
                   "argument 'probs' or argument 'n' to be set."
            
            # FIXME: give more thought about how to randomize the probability distribution
            before_normalization = np.random.rand(n)
            
        else:
            if not_normalized is None:
               not_normalized = False

            if isinstance(probs, numbers.Number):
                probs = [probs]

            if n is None:
                n = len(probs)
            else:            
                assert n == len(probs), \
                       "Constructor of ProbabilityDistribution received " \
                       "argument 'probs' with length " + str(len(probs)) + \
                       " and argument 'n' equal to " + str(n) + ", " \
                       "whereas it expects these two quntities to be equal to each other."                   
            
            before_normalization = np.array(probs, dtype=float)
            
            assert before_normalization.ndim == 1, \
                   'Constructor of ProbabilityDistribution received a vector of ' + \
                   str(before_normalization.ndim) + 'dimensions, \
                   whereas it expects exactly 1 dimension.'

        # math.fsum is probably more stable than np.sum
        if not_normalized == True:
            self.probs = before_normalization / math.fsum(before_normalization)
        else:
            self.probs = before_normalization

        assert is_close(1, math.fsum(self.probs)), \
               'Probability vector is not normalized'


    def __len__(self):
        return len(self.probs)
            


# ** QuantumState **
class QuantumState:
    """A vector of complex numbers whose norm is 1"""

    def __init__(self, amplitudes = None, nqubits = None, not_normalized = None):
        """
        If amplitudes is specified then this is the amplitudes vector used for initialization,
        otherwise we create a random quantum state of length 2**nqubits.
        By default, the amplitudes vector is required to be normalized.
        There is an option to set not_normalized to false,
        in which case we normalize the amplitudes vector.
        """

        if amplitudes is None:
           # Will generate a random quantum state for the given number of qubits

           assert nqubits is not None, \
                  "Constructor of QuantumState: argument 'nqubits' cannot be None if argument 'amplitudes' is None"

           assert not_normalized is None, \
                  "Constructor of QuantumState: " \
                  "argument 'not_normalized' is irrelevant " \
                  "if argument 'amplitudes' is None"

           assert float(nqubits).is_integer() and nqubits >= 1, \
                  'A request to generate a quantum state with ' + str(n) + ' qubits. ' \
                  'Number of qubits must be an integer strictly greater than 0.'

           not_normalized = True

           # FIXME: give more thought about how to randomize the amplitudes
           before_normalization = randcomplex(2**nqubits)

        else:
           if not_normalized is None:
               not_normalized = False

           if nqubits is None:
               nqubits = math.log2(len(amplitudes))
               assert nqubits.is_integer(), \
                      'Constructor of QuantumState received an amplitudes vector ' \
                      'of length ' + len(amplitudes) + \
                      ', whereas the number of amplitudes must be a power of 2.'
           else:           
               assert nqubits == math.log2(len(amplitudes)), \
                      'Constructor of QuantumState received ' + \
                      str(len(amplitudes)) + ' amplitudes for ' + \
                      str(nqubits) + ' qubits, ' \
                      'whereas it expects the number of amplitudes to be 2 to the power of the number of qubits.'

           before_normalization = np.array(amplitudes, dtype=complex)

           assert before_normalization.ndim == 1, \
                  'Constructor of QuantumState received a vector of ' + \
                  str(before_normalization.ndim) + 'dimensions, \
                  whereas it expects exactly 1 dimension.'

        if not_normalized == True:
            self.amplitudes = before_normalization / np.linalg.norm(before_normalization)
        else:
            self.amplitudes = before_normalization

        assert is_close(1, np.linalg.norm(self.amplitudes)), \
               'Quantum state is not normalized'
       
        self.nqubits = nqubits
        self.nstates = 2**nqubits


    @staticmethod
    def basic_state(state):
        """
        Creates a single qubit state.
        If state=0 then the created state is the ground state.
        If state=1 then the created state is the excited state.
        """
        amplitudes = np.zeros(2)
        amplitudes[state] = 1
        return QuantumState(amplitudes = amplitudes)

      

# ** DensityMatrix **
class DensityMatrix:

    def __init__(self, states = None, probs = None, mat = None):
        """
        Initialized either by a density matrix mat,
        which is required to be positive and have trace 1,
        or by a set of states with a correspoding set of probabilities for each state
        """

        # FIXME: accept a mixture of quantum states and density matrices,
        # recursively apply for all of them,
        # and then sum with the given weights.

        if mat is not None:
            assert states is None and probs is None, \
                   "Constructor of DensityMatrix: " \
                   "If argument 'mat' is not None " \
                   "then arguments 'states' and 'probs' must be set to None"

            self.rho = np.array(mat, dtype=complex)

        else:
            assert states is not None, \
                   "Constructor of DensityMatrix: " \
                   "If argument 'mat' is None " \
                   "then argument 'states' cannot be set to None" 
            
            if isinstance(states, QuantumState):
                states = [states]

            if probs is None:
                probs = ProbabilityDistribution(1)
                
            assert isinstance(states, list) and all(isinstance(s, QuantumState) for s in states) , \
                   'Constructor of DensityMatrix expects a list of QuantumState'

            assert isinstance(probs, ProbabilityDistribution), \
                   'Constructor of DensityMatrix expects a probability vector'

            assert len(probs) == len(states), \
                   'Constructor of DensityMatrix received ' \
                   'a state vector of length ' + str(len(states)) + \
                   ' and a probability distribution of length ' + str(len(probs)) + \
                   ', whereas it expects both to be of the same length.'

            assert all(s.nqubits == states[0].nqubits for s in states), \
                   'Constructor of DensityMatrix expects all quantum states to have the same number of qubits'

            mats = [np.outer(x, np.conj(x)) for x in [s.amplitudes for s in states]]
            self.rho = arraydot(probs.probs, mats)

        assert len(self.rho) == len(self.rho[0]), \
               'Constructor of DensityMatrix received a matrix with ' + \
               len(self.rho) + ' rows and ' + len(self.rho[0]) + 'columns, ' \
               'whereas it expects a sqaured matrix.'

        self.nstates = len(self.rho)
        self.nqubits = math.log2(self.nstates)

        assert self.nqubits.is_integer(), \
               'Constructor of DensityMatrix received a matrix with ' + \
               self.nstates + ' rows and columns, ' \
               'whereas the number of rows and columns must be a power of 2.'

        self.nqubits = int(self.nqubits)

        assert is_close(1, np.trace(self.rho)), \
               'Constructor of DensityMatrix received a matrix with ' \
               'trace ' + str(np.trace(self.rho)) + \
               ', whereas the trace must be equal to 1.'

        assert np.all(np.linalg.eigvals(self.rho) >= -0.05), \
               'Constructor of DensityMatrix expects a positive matrix.'


    def qop(self, operators):
        """
        Apply a set of operators on the density matrix, resulting in a new density matrix
        rho' = sum_k E_k rho E_k^\dagger
        """

        # FIXME: add an option to provide the operators as a set of unitary matrices with weights.
        
        return DensityMatrix(mat = sum( [np.dot(op, np.dot(self.rho, np.matrix(op).H)) for op in operators] ))
        

    def single_qubit_noise(self, qubit, operators):
        """
        Apply a set of operators on the specified qubit
        """

        # FIXME: generalize to multiple qubits noise
        
        return self.qop([np.kron(np.identity(2**(self.nqubits-qubit-1)), np.kron(op, np.identity(2**qubit))) for op in operators])



# ** UnitaryOperation **
class UnitaryOperation:

    def __init__(self, mat = None, beta = None, gamma = None, delta = None):
        """        
        Initialized either by a matrix mat,
        which is required to be unitary,
        or by the angles beta, gamma, and delta,
        in which case a 2x2 unitary matrix is generated.

        Generates random angles for None angles.
        I.e., generates a random 2x2 unitary matrix
        if the user does not specify any arguments to the constructor.
        """

        if mat is not None:
            assert beta is None and gamma is None and delta is None, \
                   "Constructor of UnitaryOperation: " \
                   "If argument 'mat' is not None " \
                   "then arguments 'beta', 'gamma', and 'delta' must be set to None"

            self.mat = np.array(mat, dtype=complex)

        else:

            if beta is None:
                beta = np.random.rand(1)[0]*math.pi

            if gamma is None:
                gamma = np.random.rand(1)[0]*math.pi

            if delta is None:
                delta = np.random.rand(1)[0]*math.pi

            mat_beta = np.array([[phase(-beta/2), 0], [0, phase(beta/2)]])
            mat_gamma = np.array([[math.cos(gamma/2), -math.sin(gamma/2)], [math.sin(gamma/2), math.cos(gamma/2)]])
            mat_delta = np.array([[phase(-delta/2), 0], [0, phase(delta/2)]])

            self.mat = np.dot(mat_beta, np.dot(mat_gamma, mat_delta))

        assert is_close_to_id(np.dot(np.matrix(self.mat).H, self.mat)), \
               'Constructor of UnitaryOperation: matrix is not unitary'

        
            
            
