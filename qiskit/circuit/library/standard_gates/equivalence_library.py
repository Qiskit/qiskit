# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Standard gates."""


# pylint: disable=invalid-name
import warnings
from qiskit.qasm import pi
from qiskit.circuit import EquivalenceLibrary, Parameter, QuantumCircuit, QuantumRegister

from qiskit.quantum_info.synthesis.ion_decompose import cnot_rxx_decompose

from . import (
    HGate,
    CHGate,
    MSGate,
    PhaseGate,
    CPhaseGate,
    RGate,
    RCCXGate,
    RXGate,
    CRXGate,
    RXXGate,
    RYGate,
    CRYGate,
    RZGate,
    CRZGate,
    RZZGate,
    RZXGate,
    SGate,
    SdgGate,
    SwapGate,
    CSwapGate,
    iSwapGate,
    SXGate,
    SXdgGate,
    CSXGate,
    DCXGate,
    TGate,
    TdgGate,
    UGate,
    CUGate,
    U1Gate,
    CU1Gate,
    U2Gate,
    U3Gate,
    CU3Gate,
    XGate,
    CXGate,
    CCXGate,
    YGate,
    CYGate,
    RYYGate,
    RZXGate,
    ECRGate,
    ZGate,
    CZGate,
)


_sel = StandardEquivalenceLibrary = EquivalenceLibrary()


# Import existing gate definitions

# HGate

q = QuantumRegister(1, 'q')
def_h = QuantumCircuit(q)
def_h.append(U2Gate(0, pi), [q[0]], [])
_sel.add_equivalence(HGate(), def_h)

# CHGate

q = QuantumRegister(2, 'q')
def_ch = QuantumCircuit(q)
for inst, qargs, cargs in [
        (SGate(), [q[1]], []),
        (HGate(), [q[1]], []),
        (TGate(), [q[1]], []),
        (CXGate(), [q[0], q[1]], []),
        (TdgGate(), [q[1]], []),
        (HGate(), [q[1]], []),
        (SdgGate(), [q[1]], [])
]:
    def_ch.append(inst, qargs, cargs)
_sel.add_equivalence(CHGate(), def_ch)

# MSGate

for num_qubits in range(2, 20):
    q = QuantumRegister(num_qubits, 'q')
    theta = Parameter('theta')
    def_ms = QuantumCircuit(q)
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            def_ms.append(RXXGate(theta), [q[i], q[j]])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        _sel.add_equivalence(MSGate(num_qubits, theta), def_ms)

# PhaseGate

q = QuantumRegister(1, 'q')
theta = Parameter('theta')
phase_to_u1 = QuantumCircuit(q)
phase_to_u1.append(U1Gate(theta), [0])
_sel.add_equivalence(PhaseGate(theta), phase_to_u1)

q = QuantumRegister(1, 'q')
theta = Parameter('theta')
phase_to_u = QuantumCircuit(q)
phase_to_u.u(0, 0, theta, 0)
_sel.add_equivalence(PhaseGate(theta), phase_to_u)

# CPhaseGate

q = QuantumRegister(2, 'q')
theta = Parameter('theta')
def_cphase = QuantumCircuit(q)
def_cphase.p(theta / 2, 0)
def_cphase.cx(0, 1)
def_cphase.p(-theta / 2, 1)
def_cphase.cx(0, 1)
def_cphase.p(theta / 2, 1)
_sel.add_equivalence(CPhaseGate(theta), def_cphase)

q = QuantumRegister(2, 'q')
theta = Parameter('theta')
cphase_to_cu1 = QuantumCircuit(q)
cphase_to_cu1.append(CU1Gate(theta), [0, 1])
_sel.add_equivalence(CPhaseGate(theta), cphase_to_cu1)

# RGate

q = QuantumRegister(1, 'q')
theta = Parameter('theta')
phi = Parameter('phi')
def_r = QuantumCircuit(q)
def_r.append(U3Gate(theta, phi - pi / 2, -phi + pi / 2), [q[0]])
_sel.add_equivalence(RGate(theta, phi), def_r)

# RCCXGate

q = QuantumRegister(3, 'q')
def_rccx = QuantumCircuit(q)
for inst, qargs, cargs in [
        (HGate(), [q[2]], []),
        (TGate(), [q[2]], []),
        (CXGate(), [q[1], q[2]], []),
        (TdgGate(), [q[2]], []),
        (CXGate(), [q[0], q[2]], []),
        (TGate(), [q[2]], []),
        (CXGate(), [q[1], q[2]], []),
        (TdgGate(), [q[2]], []),
        (HGate(), [q[2]], []),
]:
    def_rccx.append(inst, qargs, cargs)
_sel.add_equivalence(RCCXGate(), def_rccx)

# RXGate

q = QuantumRegister(1, 'q')
theta = Parameter('theta')
def_rx = QuantumCircuit(q)
def_rx.append(RGate(theta, 0), [q[0]], [])
_sel.add_equivalence(RXGate(theta), def_rx)

# CRXGate

q = QuantumRegister(2, 'q')
theta = Parameter('theta')
def_crx = QuantumCircuit(q)
for inst, qargs, cargs in [
        (U1Gate(pi / 2), [q[1]], []),
        (CXGate(), [q[0], q[1]], []),
        (U3Gate(-theta / 2, 0, 0), [q[1]], []),
        (CXGate(), [q[0], q[1]], []),
        (U3Gate(theta / 2, -pi / 2, 0), [q[1]], [])
]:
    def_crx.append(inst, qargs, cargs)
_sel.add_equivalence(CRXGate(theta), def_crx)

# RXXGate

q = QuantumRegister(2, 'q')
theta = Parameter('theta')
def_rxx = QuantumCircuit(q)
for inst, qargs, cargs in [
        (HGate(), [q[0]], []),
        (HGate(), [q[1]], []),
        (CXGate(), [q[0], q[1]], []),
        (RZGate(theta), [q[1]], []),
        (CXGate(), [q[0], q[1]], []),
        (HGate(), [q[1]], []),
        (HGate(), [q[0]], []),
]:
    def_rxx.append(inst, qargs, cargs)
_sel.add_equivalence(RXXGate(theta), def_rxx)

# RZXGate

q = QuantumRegister(2, 'q')
theta = Parameter('theta')
def_rzx = QuantumCircuit(q)
for inst, qargs, cargs in [
        (HGate(), [q[1]], []),
        (CXGate(), [q[0], q[1]], []),
        (RZGate(theta), [q[1]], []),
        (CXGate(), [q[0], q[1]], []),
        (HGate(), [q[1]], []),
]:
    def_rzx.append(inst, qargs, cargs)
_sel.add_equivalence(RZXGate(theta), def_rzx)


# RYGate

q = QuantumRegister(1, 'q')
theta = Parameter('theta')
def_ry = QuantumCircuit(q)
def_ry.append(RGate(theta, pi / 2), [q[0]], [])
_sel.add_equivalence(RYGate(theta), def_ry)

# CRYGate

q = QuantumRegister(2, 'q')
theta = Parameter('theta')
def_cry = QuantumCircuit(q)
for inst, qargs, cargs in [
        (U3Gate(theta / 2, 0, 0), [q[1]], []),
        (CXGate(), [q[0], q[1]], []),
        (U3Gate(-theta / 2, 0, 0), [q[1]], []),
        (CXGate(), [q[0], q[1]], [])
]:
    def_cry.append(inst, qargs, cargs)
_sel.add_equivalence(CRYGate(theta), def_cry)

# RYYGate

q = QuantumRegister(2, 'q')
theta = Parameter('theta')
def_ryy = QuantumCircuit(q)
for inst, qargs, cargs in [
        (RXGate(pi / 2), [q[0]], []),
        (RXGate(pi / 2), [q[1]], []),
        (CXGate(), [q[0], q[1]], []),
        (RZGate(theta), [q[1]], []),
        (CXGate(), [q[0], q[1]], []),
        (RXGate(-pi / 2), [q[0]], []),
        (RXGate(-pi / 2), [q[1]], []),
]:
    def_ryy.append(inst, qargs, cargs)
_sel.add_equivalence(RYYGate(theta), def_ryy)

# RZGate

q = QuantumRegister(1, 'q')
theta = Parameter('theta')
def_rz = QuantumCircuit(q, global_phase=-theta / 2)
def_rz.append(U1Gate(theta), [q[0]], [])
_sel.add_equivalence(RZGate(theta), def_rz)

q = QuantumRegister(1, 'q')
rz_to_sxry = QuantumCircuit(q)
rz_to_sxry.sx(0)
rz_to_sxry.ry(-theta, 0)
rz_to_sxry.sxdg(0)
_sel.add_equivalence(RZGate(theta), rz_to_sxry)

# CRZGate

q = QuantumRegister(2, 'q')
theta = Parameter('theta')
def_crz = QuantumCircuit(q)
for inst, qargs, cargs in [
        (U1Gate(theta / 2), [q[1]], []),
        (CXGate(), [q[0], q[1]], []),
        (U1Gate(-theta / 2), [q[1]], []),
        (CXGate(), [q[0], q[1]], [])
]:
    def_crz.append(inst, qargs, cargs)
_sel.add_equivalence(CRZGate(theta), def_crz)

# RZZGate

q = QuantumRegister(2, 'q')
theta = Parameter('theta')
def_rzz = QuantumCircuit(q)
for inst, qargs, cargs in [
        (CXGate(), [q[0], q[1]], []),
        (RZGate(theta), [q[1]], []),
        (CXGate(), [q[0], q[1]], [])
]:
    def_rzz.append(inst, qargs, cargs)
_sel.add_equivalence(RZZGate(theta), def_rzz)

# RZXGate

q = QuantumRegister(2, 'q')
theta = Parameter('theta')
def_rzx = QuantumCircuit(q)
for inst, qargs, cargs in [
        (HGate(), [q[1]], []),
        (CXGate(), [q[0], q[1]], []),
        (RZGate(theta), [q[1]], []),
        (CXGate(), [q[0], q[1]], []),
        (HGate(), [q[1]], [])
]:
    def_rzx.append(inst, qargs, cargs)
_sel.add_equivalence(RZXGate(theta), def_rzx)

# ECRGate

q = QuantumRegister(2, 'q')
def_ecr = QuantumCircuit(q)
for inst, qargs, cargs in [
        (RZXGate(pi/4), [q[0], q[1]], []),
        (XGate(), [q[0]], []),
        (RZXGate(-pi/4), [q[0], q[1]], [])
]:
    def_ecr.append(inst, qargs, cargs)
_sel.add_equivalence(ECRGate(), def_ecr)

# SGate

q = QuantumRegister(1, 'q')
def_s = QuantumCircuit(q)
def_s.append(U1Gate(pi / 2), [q[0]], [])
_sel.add_equivalence(SGate(), def_s)

# SdgGate

q = QuantumRegister(1, 'q')
def_sdg = QuantumCircuit(q)
def_sdg.append(U1Gate(-pi / 2), [q[0]], [])
_sel.add_equivalence(SdgGate(), def_sdg)

# SwapGate

q = QuantumRegister(2, 'q')
def_swap = QuantumCircuit(q)
for inst, qargs, cargs in [
        (CXGate(), [q[0], q[1]], []),
        (CXGate(), [q[1], q[0]], []),
        (CXGate(), [q[0], q[1]], [])
]:
    def_swap.append(inst, qargs, cargs)
_sel.add_equivalence(SwapGate(), def_swap)

# iSwapGate

q = QuantumRegister(2, 'q')
def_iswap = QuantumCircuit(q)
for inst, qargs, cargs in [
        (SGate(), [q[0]], []),
        (SGate(), [q[1]], []),
        (HGate(), [q[0]], []),
        (CXGate(), [q[0], q[1]], []),
        (CXGate(), [q[1], q[0]], []),
        (HGate(), [q[1]], [])
]:
    def_iswap.append(inst, qargs, cargs)
_sel.add_equivalence(iSwapGate(), def_iswap)

# SXGate

q = QuantumRegister(1, 'q')
def_sx = QuantumCircuit(q, global_phase=pi / 4)
for inst, qargs, cargs in [
        (SdgGate(), [q[0]], []),
        (HGate(), [q[0]], []),
        (SdgGate(), [q[0]], [])
]:
    def_sx.append(inst, qargs, cargs)
_sel.add_equivalence(SXGate(), def_sx)

q = QuantumRegister(1, 'q')
sx_to_rx = QuantumCircuit(q, global_phase=pi / 4)
sx_to_rx.rx(pi / 2, 0)
_sel.add_equivalence(SXGate(), sx_to_rx)

# SXdgGate

q = QuantumRegister(1, 'q')
def_sxdg = QuantumCircuit(q, global_phase=-pi / 4)
for inst, qargs, cargs in [
        (SGate(), [q[0]], []),
        (HGate(), [q[0]], []),
        (SGate(), [q[0]], [])
]:
    def_sxdg.append(inst, qargs, cargs)
_sel.add_equivalence(SXdgGate(), def_sxdg)

q = QuantumRegister(1, 'q')
sxdg_to_rx = QuantumCircuit(q, global_phase=-pi / 4)
sxdg_to_rx.rx(-pi / 2, 0)
_sel.add_equivalence(SXdgGate(), sxdg_to_rx)

# CSXGate

q = QuantumRegister(2, 'q')
def_csx = QuantumCircuit(q)
for inst, qargs, cargs in [
        (HGate(), [q[1]], []),
        (CU1Gate(pi / 2), [q[0], q[1]], []),
        (HGate(), [q[1]], [])
]:
    def_csx.append(inst, qargs, cargs)
_sel.add_equivalence(CSXGate(), def_csx)

# DCXGate

q = QuantumRegister(2, 'q')
def_dcx = QuantumCircuit(q)
for inst, qargs, cargs in [
        (CXGate(), [q[0], q[1]], []),
        (CXGate(), [q[1], q[0]], [])
]:
    def_dcx.append(inst, qargs, cargs)
_sel.add_equivalence(DCXGate(), def_dcx)

q = QuantumRegister(2, 'q')
dcx_to_iswap = QuantumCircuit(q)
for inst, qargs, cargs in [
        (HGate(), [q[0]], []),
        (SdgGate(), [q[0]], []),
        (SdgGate(), [q[1]], []),
        (iSwapGate(), [q[0], q[1]], []),
        (HGate(), [q[1]], [])
]:
    dcx_to_iswap.append(inst, qargs, cargs)
_sel.add_equivalence(DCXGate(), dcx_to_iswap)

# CSwapGate

q = QuantumRegister(3, 'q')
def_cswap = QuantumCircuit(q)
for inst, qargs, cargs in [
        (CXGate(), [q[2], q[1]], []),
        (CCXGate(), [q[0], q[1], q[2]], []),
        (CXGate(), [q[2], q[1]], [])
]:
    def_cswap.append(inst, qargs, cargs)
_sel.add_equivalence(CSwapGate(), def_cswap)

# TGate

q = QuantumRegister(1, 'q')
def_t = QuantumCircuit(q)
def_t.append(U1Gate(pi / 4), [q[0]], [])
_sel.add_equivalence(TGate(), def_t)

# TdgGate

q = QuantumRegister(1, 'q')
def_tdg = QuantumCircuit(q)
def_tdg.append(U1Gate(-pi / 4), [q[0]], [])
_sel.add_equivalence(TdgGate(), def_tdg)

# UGate

q = QuantumRegister(1, 'q')
theta = Parameter('theta')
phi = Parameter('phi')
lam = Parameter('lam')
u_to_u3 = QuantumCircuit(q)
u_to_u3.append(U3Gate(theta, phi, lam), [0])
_sel.add_equivalence(UGate(theta, phi, lam), u_to_u3)

# CUGate

q = QuantumRegister(2, 'q')
theta = Parameter('theta')
phi = Parameter('phi')
lam = Parameter('lam')
gamma = Parameter('gamma')
def_cu = QuantumCircuit(q)
def_cu.p(gamma, 0)
def_cu.p((lam + phi) / 2, 0)
def_cu.p((lam - phi) / 2, 1)
def_cu.cx(0, 1)
def_cu.u(-theta / 2, 0, -(phi + lam) / 2, 1)
def_cu.cx(0, 1)
def_cu.u(theta / 2, phi, 0, 1)
_sel.add_equivalence(CUGate(theta, phi, lam, gamma), def_cu)

q = QuantumRegister(2, 'q')
theta = Parameter('theta')
phi = Parameter('phi')
lam = Parameter('lam')
gamma = Parameter('gamma')
cu_to_cu3 = QuantumCircuit(q)
cu_to_cu3.p(gamma, 0)
cu_to_cu3.append(CU3Gate(theta, phi, lam), [0, 1])
_sel.add_equivalence(CUGate(theta, phi, lam, gamma), cu_to_cu3)

# U1Gate

q = QuantumRegister(1, 'q')
theta = Parameter('theta')
def_u1 = QuantumCircuit(q)
def_u1.append(U3Gate(0, 0, theta), [q[0]], [])
_sel.add_equivalence(U1Gate(theta), def_u1)

q = QuantumRegister(1, 'q')
theta = Parameter('theta')
u1_to_phase = QuantumCircuit(q)
u1_to_phase.p(theta, 0)
_sel.add_equivalence(U1Gate(theta), u1_to_phase)

# CU1Gate

q = QuantumRegister(2, 'q')
theta = Parameter('theta')
def_cu1 = QuantumCircuit(q)
for inst, qargs, cargs in [
        (U1Gate(theta / 2), [q[0]], []),
        (CXGate(), [q[0], q[1]], []),
        (U1Gate(-theta / 2), [q[1]], []),
        (CXGate(), [q[0], q[1]], []),
        (U1Gate(theta / 2), [q[1]], [])
]:
    def_cu1.append(inst, qargs, cargs)
_sel.add_equivalence(CU1Gate(theta), def_cu1)

# U1Gate

q = QuantumRegister(1, 'q')
phi = Parameter('phi')
lam = Parameter('lam')
def_u2 = QuantumCircuit(q)
def_u2.append(U3Gate(pi / 2, phi, lam), [q[0]], [])
_sel.add_equivalence(U2Gate(phi, lam), def_u2)

# U2Gate

q = QuantumRegister(1, 'q')
phi = Parameter('phi')
lam = Parameter('lam')
u2_to_u1sx = QuantumCircuit(q, global_phase=-pi / 4)
u2_to_u1sx.append(U1Gate(lam - pi/2), [0])
u2_to_u1sx.sx(0)
u2_to_u1sx.append(U1Gate(phi + pi/2), [0])
_sel.add_equivalence(U2Gate(phi, lam), u2_to_u1sx)

# U3Gate

q = QuantumRegister(1, 'q')
theta = Parameter('theta')
phi = Parameter('phi')
lam = Parameter('lam')
u3_qasm_def = QuantumCircuit(q, global_phase=(lam + phi - pi) / 2)
u3_qasm_def.rz(lam, 0)
u3_qasm_def.sx(0)
u3_qasm_def.rz(theta+pi, 0)
u3_qasm_def.sx(0)
u3_qasm_def.rz(phi+3*pi, 0)
_sel.add_equivalence(U3Gate(theta, phi, lam), u3_qasm_def)

q = QuantumRegister(1, 'q')
theta = Parameter('theta')
phi = Parameter('phi')
lam = Parameter('lam')
u3_to_u = QuantumCircuit(q)
u3_to_u.u(theta, phi, lam, 0)
_sel.add_equivalence(U3Gate(theta, phi, lam), u3_to_u)

# CU3Gate

q = QuantumRegister(2, 'q')
theta = Parameter('theta')
phi = Parameter('phi')
lam = Parameter('lam')
def_cu3 = QuantumCircuit(q)
for inst, qargs, cargs in [
        (U1Gate((lam + phi) / 2), [q[0]], []),
        (U1Gate((lam - phi) / 2), [q[1]], []),
        (CXGate(), [q[0], q[1]], []),
        (U3Gate(-theta / 2, 0, -(phi + lam) / 2), [q[1]], []),
        (CXGate(), [q[0], q[1]], []),
        (U3Gate(theta / 2, phi, 0), [q[1]], [])
]:
    def_cu3.append(inst, qargs, cargs)
_sel.add_equivalence(CU3Gate(theta, phi, lam), def_cu3)

q = QuantumRegister(2, 'q')
theta = Parameter('theta')
phi = Parameter('phi')
lam = Parameter('lam')
cu3_to_cu = QuantumCircuit(q)
cu3_to_cu.cu(theta, phi, lam, 0, 0, 1)

# XGate

q = QuantumRegister(1, 'q')
def_x = QuantumCircuit(q)
def_x.append(U3Gate(pi, 0, pi), [q[0]], [])
_sel.add_equivalence(XGate(), def_x)

# CXGate

for plus_ry in [False, True]:
    for plus_rxx in [False, True]:
        cx_to_rxx = cnot_rxx_decompose(plus_ry, plus_rxx)
        _sel.add_equivalence(CXGate(), cx_to_rxx)

q = QuantumRegister(2, 'q')
cx_to_cz = QuantumCircuit(q)
for inst, qargs, cargs in [
        (HGate(), [q[1]], []),
        (CZGate(), [q[0], q[1]], []),
        (HGate(), [q[1]], [])
]:
    cx_to_cz.append(inst, qargs, cargs)
_sel.add_equivalence(CXGate(), cx_to_cz)

q = QuantumRegister(2, 'q')
cx_to_iswap = QuantumCircuit(q, global_phase=3*pi/4)
for inst, qargs, cargs in [
        (HGate(), [q[0]], []),
        (XGate(), [q[1]], []),
        (HGate(), [q[1]], []),
        (iSwapGate(), [q[0], q[1]], []),
        (XGate(), [q[0]], []),
        (XGate(), [q[1]], []),
        (HGate(), [q[1]], []),
        (iSwapGate(), [q[0], q[1]], []),
        (HGate(), [q[0]], []),
        (SGate(), [q[0]], []),
        (SGate(), [q[1]], []),
        (XGate(), [q[1]], []),
        (HGate(), [q[1]], []),
]:
    cx_to_iswap.append(inst, qargs, cargs)
_sel.add_equivalence(CXGate(), cx_to_iswap)

q = QuantumRegister(2, 'q')
cx_to_ecr = QuantumCircuit(q, global_phase=-pi/4)
for inst, qargs, cargs in [
        (RZGate(-pi/2), [q[0]], []),
        (RYGate(pi), [q[0]], []),
        (RXGate(pi/2), [q[1]], []),
        (ECRGate(), [q[0], q[1]], [])
]:
    cx_to_ecr.append(inst, qargs, cargs)
_sel.add_equivalence(CXGate(), cx_to_ecr)

# CCXGate

q = QuantumRegister(3, 'q')
def_ccx = QuantumCircuit(q)
for inst, qargs, cargs in [
        (HGate(), [q[2]], []),
        (CXGate(), [q[1], q[2]], []),
        (TdgGate(), [q[2]], []),
        (CXGate(), [q[0], q[2]], []),
        (TGate(), [q[2]], []),
        (CXGate(), [q[1], q[2]], []),
        (TdgGate(), [q[2]], []),
        (CXGate(), [q[0], q[2]], []),
        (TGate(), [q[1]], []),
        (TGate(), [q[2]], []),
        (HGate(), [q[2]], []),
        (CXGate(), [q[0], q[1]], []),
        (TGate(), [q[0]], []),
        (TdgGate(), [q[1]], []),
        (CXGate(), [q[0], q[1]], [])
]:
    def_ccx.append(inst, qargs, cargs)
_sel.add_equivalence(CCXGate(), def_ccx)

# YGate

q = QuantumRegister(1, 'q')
def_y = QuantumCircuit(q)
def_y.append(U3Gate(pi, pi / 2, pi / 2), [q[0]], [])
_sel.add_equivalence(YGate(), def_y)

# CYGate

q = QuantumRegister(2, 'q')
def_cy = QuantumCircuit(q)
for inst, qargs, cargs in [
        (SdgGate(), [q[1]], []),
        (CXGate(), [q[0], q[1]], []),
        (SGate(), [q[1]], [])
]:
    def_cy.append(inst, qargs, cargs)
_sel.add_equivalence(CYGate(), def_cy)

# ZGate

q = QuantumRegister(1, 'q')
def_z = QuantumCircuit(q)
def_z.append(U1Gate(pi), [q[0]], [])
_sel.add_equivalence(ZGate(), def_z)

# CZGate

q = QuantumRegister(2, 'q')
def_cz = QuantumCircuit(q)
for inst, qargs, cargs in [
        (HGate(), [q[1]], []),
        (CXGate(), [q[0], q[1]], []),
        (HGate(), [q[1]], [])
]:
    def_cz.append(inst, qargs, cargs)
_sel.add_equivalence(CZGate(), def_cz)

# RXGate, XGate equivalence

q = QuantumRegister(1, 'q')
x_to_rx = QuantumCircuit(q)
x_to_rx.append(RXGate(theta=pi), [q[0]])
x_to_rx.global_phase = pi/2
_sel.add_equivalence(XGate(), x_to_rx)

# RYGate, YGate equivalence

q = QuantumRegister(1, 'q')
y_to_ry = QuantumCircuit(q)
y_to_ry.append(RYGate(theta=pi), [q[0]])
y_to_ry.global_phase = pi/2
_sel.add_equivalence(YGate(), y_to_ry)


# HGate, RXGate(pi).RYGate(pi/2) equivalence

q = QuantumRegister(1, 'q')
h_to_rxry = QuantumCircuit(q)
h_to_rxry.append(RYGate(theta=pi/2), [q[0]])
h_to_rxry.append(RXGate(theta=pi), [q[0]])
h_to_rxry.global_phase = pi/2
_sel.add_equivalence(HGate(), h_to_rxry)

# HGate, RGate(pi, 0).RGate(pi/2, pi/2) equivalence

q = QuantumRegister(1, 'q')
h_to_rr = QuantumCircuit(q)
h_to_rr.append(RGate(theta=pi/2, phi=pi/2), [q[0]])
h_to_rr.append(RGate(theta=pi, phi=0), [q[0]])
h_to_rr.global_phase = pi/2
_sel.add_equivalence(HGate(), h_to_rr)
