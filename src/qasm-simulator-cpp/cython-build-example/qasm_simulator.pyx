from libcpp.string cimport string

# Link against SciPy BLAS functions
from scipy.linalg.cython_blas cimport sgemv, dgemv, cgemv, zgemv
from scipy.linalg.cython_blas cimport sgemm, dgemm, cgemm, zgemm

# Import C++ Simulator class
cdef extern from "simulator.hpp" namespace "QISKIT":

    cdef cppclass Simulator:
        Simulator() except +
        string execute(int indent) except +
        void load_qobj_string(string qobj) except +


cdef class SimulatorWrapper:
    """
    Python wrapper of C++ Simulator class.

    Methods:
        run: executes a qobj.
    """
    cdef Simulator *thisptr

    def __cinit__(self):
        self.thisptr = new Simulator()

    def __dealloc__(self):
        del self.thisptr

    def run(self, qobj_str):
        """
        Execute a qobj

        Args:
            qobj_str (str): a qobj JSON serialized as a python string.

        Returns:
            result JSON serialized as a python string.
        """
        # serialize qobj as json byte string
        self.thisptr.load_qobj_string(qobj_str.encode())
        return self.thisptr.execute(-1).decode()
