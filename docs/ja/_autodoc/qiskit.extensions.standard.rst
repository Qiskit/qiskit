qiskit.extensions.standard package
==================================


.. automodule:: qiskit.extensions.standard
    
    

    Classes
    -------


    .. list-table::
    
       * - :class:`Barrier <qiskit.extensions.standard.barrier.Barrier>`
         - Barrier instruction.
       * - :class:`CHGate <qiskit.extensions.standard.ch.CHGate>`
         - controlled-H gate.
       * - :class:`CXBase <qiskit.extensions.standard.cxbase.CXBase>`
         - Fundamental controlled-NOT gate.
       * - :class:`CnotGate <qiskit.extensions.standard.cx.CnotGate>`
         - controlled-NOT gate.
       * - :class:`CrzGate <qiskit.extensions.standard.crz.CrzGate>`
         - controlled-rz gate.
       * - :class:`Cu1Gate <qiskit.extensions.standard.cu1.Cu1Gate>`
         - controlled-u1 gate.
       * - :class:`Cu3Gate <qiskit.extensions.standard.cu3.Cu3Gate>`
         - controlled-u3 gate.
       * - :class:`CyGate <qiskit.extensions.standard.cy.CyGate>`
         - controlled-Y gate.
       * - :class:`CzGate <qiskit.extensions.standard.cz.CzGate>`
         - controlled-Z gate.
       * - :class:`FredkinGate <qiskit.extensions.standard.cswap.FredkinGate>`
         - Fredkin gate.
       * - :class:`HGate <qiskit.extensions.standard.h.HGate>`
         - Hadamard gate.
       * - :class:`IdGate <qiskit.extensions.standard.iden.IdGate>`
         - Identity gate.
       * - :class:`RXGate <qiskit.extensions.standard.rx.RXGate>`
         - rotation around the x-axis.
       * - :class:`RYGate <qiskit.extensions.standard.ry.RYGate>`
         - rotation around the y-axis.
       * - :class:`RZGate <qiskit.extensions.standard.rz.RZGate>`
         - rotation around the z-axis.
       * - :class:`RZZGate <qiskit.extensions.standard.rzz.RZZGate>`
         - Two-qubit ZZ-rotation gate.
       * - :class:`SGate <qiskit.extensions.standard.s.SGate>`
         - S=diag(1,i) Clifford phase gate.
       * - :class:`SdgGate <qiskit.extensions.standard.s.SdgGate>`
         - Sdg=diag(1,-i) Clifford adjoin phase gate.
       * - :class:`SwapGate <qiskit.extensions.standard.swap.SwapGate>`
         - SWAP gate.
       * - :class:`TGate <qiskit.extensions.standard.t.TGate>`
         - T Gate: pi/4 rotation around Z axis.
       * - :class:`TdgGate <qiskit.extensions.standard.t.TdgGate>`
         - T Gate: -pi/4 rotation around Z axis.
       * - :class:`ToffoliGate <qiskit.extensions.standard.ccx.ToffoliGate>`
         - Toffoli gate.
       * - :class:`U0Gate <qiskit.extensions.standard.u0.U0Gate>`
         - Wait gate.
       * - :class:`U1Gate <qiskit.extensions.standard.u1.U1Gate>`
         - Diagonal single-qubit gate.
       * - :class:`U2Gate <qiskit.extensions.standard.u2.U2Gate>`
         - One-pulse single-qubit gate.
       * - :class:`U3Gate <qiskit.extensions.standard.u3.U3Gate>`
         - Two-pulse single-qubit gate.
       * - :class:`UBase <qiskit.extensions.standard.ubase.UBase>`
         - Element of SU(2).
       * - :class:`XGate <qiskit.extensions.standard.x.XGate>`
         - Pauli X (bit-flip) gate.
       * - :class:`YGate <qiskit.extensions.standard.y.YGate>`
         - Pauli Y (bit-phase-flip) gate.
       * - :class:`ZGate <qiskit.extensions.standard.z.ZGate>`
         - Pauli Z (phase-flip) gate.
    




    Functions
    ---------


    .. list-table::
    
       * - :func:`barrier <qiskit.extensions.standard.barrier.barrier>`
         - Apply barrier to circuit.
       * - :func:`ccx <qiskit.extensions.standard.ccx.ccx>`
         - Apply Toffoli to from ctl1 and ctl2 to tgt.
       * - :func:`ch <qiskit.extensions.standard.ch.ch>`
         - Apply CH from ctl to tgt.
       * - :func:`crz <qiskit.extensions.standard.crz.crz>`
         - Apply crz from ctl to tgt with angle theta.
       * - :func:`cswap <qiskit.extensions.standard.cswap.cswap>`
         - Apply Fredkin to circuit.
       * - :func:`cu1 <qiskit.extensions.standard.cu1.cu1>`
         - Apply cu1 from ctl to tgt with angle theta.
       * - :func:`cu3 <qiskit.extensions.standard.cu3.cu3>`
         - Apply cu3 from ctl to tgt with angle theta, phi, lam.
       * - :func:`cx <qiskit.extensions.standard.cx.cx>`
         - Apply CX from ctl to tgt.
       * - :func:`cx_base <qiskit.extensions.standard.cxbase.cx_base>`
         - Apply CX ctl, tgt.
       * - :func:`cy <qiskit.extensions.standard.cy.cy>`
         - Apply CY to circuit.
       * - :func:`cz <qiskit.extensions.standard.cz.cz>`
         - Apply CZ to circuit.
       * - :func:`h <qiskit.extensions.standard.h.h>`
         - Apply H to q.
       * - :func:`iden <qiskit.extensions.standard.iden.iden>`
         - Apply Identity to q.
       * - :func:`rx <qiskit.extensions.standard.rx.rx>`
         - Apply Rx to q.
       * - :func:`ry <qiskit.extensions.standard.ry.ry>`
         - Apply Ry to q.
       * - :func:`rz <qiskit.extensions.standard.rz.rz>`
         - Apply Rz to q.
       * - :func:`rzz <qiskit.extensions.standard.rzz.rzz>`
         - Apply RZZ to circuit.
       * - :func:`s <qiskit.extensions.standard.s.s>`
         - Apply S to q.
       * - :func:`sdg <qiskit.extensions.standard.s.sdg>`
         - Apply Sdg to q.
       * - :func:`swap <qiskit.extensions.standard.swap.swap>`
         - Apply SWAP from ctl to tgt.
       * - :func:`t <qiskit.extensions.standard.t.t>`
         - Apply T to q.
       * - :func:`tdg <qiskit.extensions.standard.t.tdg>`
         - Apply Tdg to q.
       * - :func:`u0 <qiskit.extensions.standard.u0.u0>`
         - Apply u0 with length m to q.
       * - :func:`u1 <qiskit.extensions.standard.u1.u1>`
         - Apply u1 with angle theta to q.
       * - :func:`u2 <qiskit.extensions.standard.u2.u2>`
         - Apply u2 to q.
       * - :func:`u3 <qiskit.extensions.standard.u3.u3>`
         - Apply u3 to q.
       * - :func:`u_base <qiskit.extensions.standard.ubase.u_base>`
         - Apply U to q.
       * - :func:`x <qiskit.extensions.standard.x.x>`
         - Apply X to q.
       * - :func:`y <qiskit.extensions.standard.y.y>`
         - Apply Y to q.
       * - :func:`z <qiskit.extensions.standard.z.z>`
         - Apply Z to q.
    