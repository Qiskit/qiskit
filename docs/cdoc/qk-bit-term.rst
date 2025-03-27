=========
QkBitTerm
=========

..
    This is documented manually here because the C-space `enum` is generated
    programmatically from Rust and is not the correct C-level documentation.

.. code-block:: c

   enum QkBitTerm

An enum that to represent each of the single-qubit alphabet terms enumerated below. 

Members
-------

  =======================================  ==================  ===============
  Operator                                 ``QkBitTerm``       Numeric value
  =======================================  ==================  ===============
  :math:`X` (Pauli X)                      ``QkBitTerm_X``     ``0b0010`` (2)   

  :math:`Y` (Pauli Y)                      ``QkBitTerm_Y``     ``0b0011`` (3)   

  :math:`Z` (Pauli Z)                      ``QkBitTerm_Z``     ``0b0001`` (1)   

  :math:`\lvert+\rangle\langle+\rvert`     ``QkBitTerm_Plus``  ``0b1010`` (10)  
  (projector to positive eigenstate of X)

  :math:`\lvert-\rangle\langle-\rvert`     ``QkBitTerm_Minus`` ``0b0110`` (6)   
  (projector to negative eigenstate of X)

  :math:`\lvert r\rangle\langle r\rvert`   ``QkBitTerm_Right`` ``0b1011`` (11)  
  (projector to positive eigenstate of Y)

  :math:`\lvert l\rangle\langle l\rvert`   ``QkBitTerm_Left``  ``0b0111`` (7)   
  (projector to negative eigenstate of Y)

  :math:`\lvert0\rangle\langle0\rvert`     ``QkBitTerm_Zero``  ``0b1001`` (9)   
  (projector to positive eigenstate of Z)

  :math:`\lvert1\rangle\langle1\rvert`     ``QkBitTerm_One``   ``0b0101`` (5)   
  (projector to negative eigenstate of Z)
  =======================================  ==================  ===============


Representation
--------------

The enum is stored as single byte, its elements are represented as unsigned 8-bit integer.

.. code-block:: c

   typedef uint8_t QkBitTerm

.. warning:: 

   Not all ``uint8_t`` values are valid bit terms. Passing invalid values is undefined behavior.

The numeric structure of these is that they are all four-bit values of which the low two
bits are the (phase-less) symplectic representation of the Pauli operator related to the
object, where the low bit denotes a contribution by :math:`Z` and the second lowest a
contribution by :math:`X`, while the upper two bits are ``00`` for a Pauli operator, ``01``
for the negative-eigenstate projector, and ``10`` for the positive-eigenstate projector.

---------
Functions
---------

.. doxygengroup:: QkBitTerm
   :content-only:
