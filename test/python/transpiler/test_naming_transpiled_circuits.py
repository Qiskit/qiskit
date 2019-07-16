# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 15:26:51 2019

@author: Eesh Gupta
"""

import unittest
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.compiler import transpile
from qiskit import BasicAer
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.test import QiskitTestCase


class test_naming_transpiled_circuits(QiskitTestCase):
    """
    Testing the naming fuctionality for transpiled circuits. 
    """
    def no_name(self): 
        """
        If no arg for output_names given for transpiled circuits, then 
        transpiled circuit assumes the name of the original circuit. This 
        test checks that condition.
        """
        #single circuit
        qr=QuantumRegister(4)
        cirkie=QuantumCircuit(qr, name='cirkie')
        qr.ccx(qr[0],qr[1],qr[2])
        basis_gates_var=['u1','u2','u3','cx']
        trans_cirkie=transpile(cirkie, basis_gates=basis_gates_var)
        self.assertTrue(trans_cirkie.name, 'cirkie')
        
        #multiple Circuits
        qr1=QuantumRegister(2)
        cr1=ClassicalRegister(2)
        circ1=QuantumCircuit(qr1,cr1,name='circ1-original')
        circ1.h(qr1[0])
        circ1.cx(qr1[0],qr1[1])
        
        qr2=QuantumRegister(2)
        cr2=ClassicalRegister(2)
        circ2=QuantumCircuit(qr2,cr2,name='circ2-original')
        circ2.measure(qr2,cr2)
        
        backendie=BasicAer.get_backend('qasm_simulator')
        trans_circuits=transpile([circ1, circ2], backend=backendie)
        self.assertTrue(trans_circuits[0].name,'circ1-original')
        self.assertTrue(trans_circuits[1].name,'circ2-original')
        
        
    def single_circuit_singleton_name (self):
        """
        Given a single circuit and a output name in form of a string, this test
        checks whether that string name is assigned to the transpiled circuit.
        """
        qr=QuantumRegister(4)
        cirkie=QuantumCircuit(qr)
        qr.ccx(qr[0],qr[1],qr[2])
        basis_gates_var=['u1','u2','u3','cx']
        trans_cirkie=transpile(cirkie, basis_gates=basis_gates_var,
                               output_names='transpiled-cirkie')
        self.assertTrue(trans_cirkie.name, 'transpiled-cirkie')
    
    def single_circuit_name_list(self): 
        """
        Given a single circuit and an output name in form of a single element
        list, this test checks whether the transpiled circuit is mapped with 
        that assigned name in the list.
        
        If list has more than one element, then test checks whether the 
        Transpile function raises an error.
        """
        qr=QuantumRegister(4)
        cirkie=QuantumCircuit(qr)
        qr.ccx(qr[0],qr[1],qr[2])
        basis_gates_var=['u1','u2','u3','cx']
        trans_cirkie=transpile(cirkie, basis_gates=basis_gates_var,
                               output_names=['transpiled-cirkie'])
        self.assertTrue(trans_cirkie.name, 'transpiled-cirkie')
        #If List has multiple elements, transpile function must raise error
        with self.assertRaises(TranspilerError): 
             transpile(cirkie, basis_gates=basis_gates_var, output_names=
                       ["transpiled-cirkie","new-cirkie","dope-cirkie",
                        "awesome-cirkie"])
    
    def multiple_circuits_name_singleton(self): 
        """
        Given multiple circuits and a single string as a name, this test checks
        whether the Transpile function raises an error.
        """
        qr1=QuantumRegister(2)
        cr1=ClassicalRegister(2)
        circ1=QuantumCircuit(qr1,cr1)
        circ1.h(qr1[0])
        circ1.cx(qr1[0],qr1[1])
        
        qr2=QuantumRegister(2)
        cr2=ClassicalRegister(2)
        circ2=QuantumCircuit(qr2,cr2)
        circ2.measure(qr2,cr2)
        
        backendie=BasicAer.get_backend('qasm_simulator')
        #Raise Error if single name given to multiple circuits
        with self.assertRaises(TranspilerError):
            transpile([circ1,circ2], backendie, output_names='circ')
            
    def multiple_circuits_name_list(self): 
        """
        Given multiple circuits and a list for output names, if 
        len(list)=len(circuits), then test checks whether transpile func assigns
        each element in list to respective circuit.
        If lengths are not equal, then test checks whether transpile func raises
        error.
        """
        qr1=QuantumRegister(2)
        cr1=ClassicalRegister(2)
        circ1=QuantumCircuit(qr1,cr1)
        circ1.h(qr1[0])
        circ1.cx(qr1[0],qr1[1])
        
        qr2=QuantumRegister(2)
        cr2=ClassicalRegister(2)
        circ2=QuantumCircuit(qr2,cr2)
        circ2.measure(qr2,cr2)
        
        qr3=QuantumRegister(2)
        cr3=ClassicalRegister(2)
        circ3=QuantumCircuit(qr3,cr3)
        circ3.x(qr3[0])
        circ3.x(qr3[1])
        
        circuits=[circ1,circ2,circ3]
        backendie=BasicAer.get_backend('qasm_simulator')
        
        #equal lengths
        names=['awesome_circ1','awesome_circ2','awesome_circ3']
        trans_circuits=transpile(circuits, backendie, output_names=names)
        self.assertTrue(trans_circuits[0].name,'awesome-circ1')
        self.assertTrue(trans_circuits[1].name,'awesome-circ2')
        self.assertTrue(trans_circuits[2].name,'awesome-circ3')
        #names list greater than circuits list
        names=['awesome_circ1','awesome_circ2','awesome_circ3','awesome-circ4']
        with self.assertRaises(TranspilerError):
             transpile(circuits, backendie, output_names=names)
        #names list smaller than circuits list
        names=['awesome_circ1','awesome_circ2']
        with self.assertRaises(TranspilerError):
             transpile(circuits, backendie, output_names=names)
if __name__ == '__main__':
    unittest.main()
           
        
        
        