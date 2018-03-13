from libcpp.string cimport string

cdef extern from "simulator.hpp" namespace "QISKIT":

    cdef cppclass Simulator:
        Simulator()

        string execute(int indent)
        void load_string(string qobj)


cdef class SimulatorWrapper:
    cdef Simulator *thisptr

    def __cinit__(self):
        self.thisptr = new Simulator()

    def __dealloc__(self):
        del self.thisptr

    def run(self, qobj_str, indent=4):
        ""
        # serialize qobj as json byte string
        self.thisptr.load_string(qobj_str.encode())
        return self.thisptr.execute(indent).decode()
