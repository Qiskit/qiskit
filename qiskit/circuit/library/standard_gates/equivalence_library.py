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


import warnings
from qiskit.qasm import pi
from qiskit.circuit import EquivalenceLibrary, Parameter, QuantumCircuit

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
    ECRGate,
    ZGate,
    CZGate,
)


_sel = StandardEquivalenceLibrary = EquivalenceLibrary()


# Import existing gate definitions

# HGate

def_h = QuantumCircuit(1)
def_h.append(U2Gate(0, pi), [0], [])
_sel.add_equivalence(HGate(), def_h)

# CHGate

def_ch = QuantumCircuit(2)
for inst, qargs, cargs in [
    (SGate(), [1], []),
    (HGate(), [1], []),
    (TGate(), [1], []),
    (CXGate(), [0, 1], []),
    (TdgGate(), [1], []),
    (HGate(), [1], []),
    (SdgGate(), [1], []),
]:
    def_ch.append(inst, qargs, cargs)
_sel.add_equivalence(CHGate(), def_ch)

# MSGate

for num_qubits in range(2, 20):
    theta = Parameter("theta")
    def_ms = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            def_ms.append(RXXGate(theta), [i, j])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        _sel.add_equivalence(MSGate(num_qubits, theta), def_ms)

# PhaseGate

theta = Parameter("theta")
phase_to_u1 = QuantumCircuit(1)
phase_to_u1.append(U1Gate(theta), [0])
_sel.add_equivalence(PhaseGate(theta), phase_to_u1)

theta = Parameter("theta")
phase_to_u = QuantumCircuit(1)
phase_to_u.u(0, 0, theta, 0)
_sel.add_equivalence(PhaseGate(theta), phase_to_u)

# CPhaseGate

theta = Parameter("theta")
def_cphase = QuantumCircuit(2)
def_cphase.p(theta / 2, 0)
def_cphase.cx(0, 1)
def_cphase.p(-theta / 2, 1)
def_cphase.cx(0, 1)
def_cphase.p(theta / 2, 1)
_sel.add_equivalence(CPhaseGate(theta), def_cphase)

theta = Parameter("theta")
cphase_to_cu1 = QuantumCircuit(2)
cphase_to_cu1.append(CU1Gate(theta), [0, 1])
_sel.add_equivalence(CPhaseGate(theta), cphase_to_cu1)

# RGate

theta = Parameter("theta")
phi = Parameter("phi")
def_r = QuantumCircuit(1)
def_r.append(U3Gate(theta, phi - pi / 2, -phi + pi / 2), [0])
_sel.add_equivalence(RGate(theta, phi), def_r)

# RCCXGate

def_rccx = QuantumCircuit(3)
for inst, qargs, cargs in [
    (HGate(), [2], []),
    (TGate(), [2], []),
    (CXGate(), [1, 2], []),
    (TdgGate(), [2], []),
    (CXGate(), [0, 2], []),
    (TGate(), [2], []),
    (CXGate(), [1, 2], []),
    (TdgGate(), [2], []),
    (HGate(), [2], []),
]:
    def_rccx.append(inst, qargs, cargs)
_sel.add_equivalence(RCCXGate(), def_rccx)

# RXGate

theta = Parameter("theta")
def_rx = QuantumCircuit(1)
def_rx.append(RGate(theta, 0), [0], [])
_sel.add_equivalence(RXGate(theta), def_rx)

# CRXGate

theta = Parameter("theta")
def_crx = QuantumCircuit(2)
for inst, qargs, cargs in [
    (U1Gate(pi / 2), [1], []),
    (CXGate(), [0, 1], []),
    (U3Gate(-theta / 2, 0, 0), [1], []),
    (CXGate(), [0, 1], []),
    (U3Gate(theta / 2, -pi / 2, 0), [1], []),
]:
    def_crx.append(inst, qargs, cargs)
_sel.add_equivalence(CRXGate(theta), def_crx)

theta = Parameter("theta")
crx_to_srycx = QuantumCircuit(2)
for inst, qargs, cargs in [
    (SGate(), [1], []),
    (CXGate(), [0, 1], []),
    (RYGate(-theta / 2), [1], []),
    (CXGate(), [0, 1], []),
    (RYGate(theta / 2), [1], []),
    (SdgGate(), [1], []),
]:
    crx_to_srycx.append(inst, qargs, cargs)
_sel.add_equivalence(CRXGate(theta), crx_to_srycx)

# RXXGate

theta = Parameter("theta")
def_rxx = QuantumCircuit(2)
for inst, qargs, cargs in [
    (HGate(), [0], []),
    (HGate(), [1], []),
    (CXGate(), [0, 1], []),
    (RZGate(theta), [1], []),
    (CXGate(), [0, 1], []),
    (HGate(), [1], []),
    (HGate(), [0], []),
]:
    def_rxx.append(inst, qargs, cargs)
_sel.add_equivalence(RXXGate(theta), def_rxx)

# RZXGate

theta = Parameter("theta")
def_rzx = QuantumCircuit(2)
for inst, qargs, cargs in [
    (HGate(), [1], []),
    (CXGate(), [0, 1], []),
    (RZGate(theta), [1], []),
    (CXGate(), [0, 1], []),
    (HGate(), [1], []),
]:
    def_rzx.append(inst, qargs, cargs)
_sel.add_equivalence(RZXGate(theta), def_rzx)


# RYGate

theta = Parameter("theta")
def_ry = QuantumCircuit(1)
def_ry.append(RGate(theta, pi / 2), [0], [])
_sel.add_equivalence(RYGate(theta), def_ry)

# CRYGate

theta = Parameter("theta")
def_cry = QuantumCircuit(2)
for inst, qargs, cargs in [
    (RYGate(theta / 2), [1], []),
    (CXGate(), [0, 1], []),
    (RYGate(-theta / 2), [1], []),
    (CXGate(), [0, 1], []),
]:
    def_cry.append(inst, qargs, cargs)
_sel.add_equivalence(CRYGate(theta), def_cry)

# RYYGate

theta = Parameter("theta")
def_ryy = QuantumCircuit(2)
for inst, qargs, cargs in [
    (RXGate(pi / 2), [0], []),
    (RXGate(pi / 2), [1], []),
    (CXGate(), [0, 1], []),
    (RZGate(theta), [1], []),
    (CXGate(), [0, 1], []),
    (RXGate(-pi / 2), [0], []),
    (RXGate(-pi / 2), [1], []),
]:
    def_ryy.append(inst, qargs, cargs)
_sel.add_equivalence(RYYGate(theta), def_ryy)

# RZGate

theta = Parameter("theta")
def_rz = QuantumCircuit(1, global_phase=-theta / 2)
def_rz.append(U1Gate(theta), [0], [])
_sel.add_equivalence(RZGate(theta), def_rz)

rz_to_sxry = QuantumCircuit(1)
rz_to_sxry.sx(0)
rz_to_sxry.ry(-theta, 0)
rz_to_sxry.sxdg(0)
_sel.add_equivalence(RZGate(theta), rz_to_sxry)

# CRZGate

theta = Parameter("theta")
def_crz = QuantumCircuit(2)
for inst, qargs, cargs in [
    (RZGate(theta / 2), [1], []),
    (CXGate(), [0, 1], []),
    (RZGate(-theta / 2), [1], []),
    (CXGate(), [0, 1], []),
]:
    def_crz.append(inst, qargs, cargs)
_sel.add_equivalence(CRZGate(theta), def_crz)

# RZZGate

theta = Parameter("theta")
def_rzz = QuantumCircuit(2)
for inst, qargs, cargs in [
    (CXGate(), [0, 1], []),
    (RZGate(theta), [1], []),
    (CXGate(), [0, 1], []),
]:
    def_rzz.append(inst, qargs, cargs)
_sel.add_equivalence(RZZGate(theta), def_rzz)

# RZXGate

theta = Parameter("theta")
def_rzx = QuantumCircuit(2)
for inst, qargs, cargs in [
    (HGate(), [1], []),
    (CXGate(), [0, 1], []),
    (RZGate(theta), [1], []),
    (CXGate(), [0, 1], []),
    (HGate(), [1], []),
]:
    def_rzx.append(inst, qargs, cargs)
_sel.add_equivalence(RZXGate(theta), def_rzx)

# ECRGate

def_ecr = QuantumCircuit(2)
for inst, qargs, cargs in [
    (RZXGate(pi / 4), [0, 1], []),
    (XGate(), [0], []),
    (RZXGate(-pi / 4), [0, 1], []),
]:
    def_ecr.append(inst, qargs, cargs)
_sel.add_equivalence(ECRGate(), def_ecr)

# SGate

def_s = QuantumCircuit(1)
def_s.append(U1Gate(pi / 2), [0], [])
_sel.add_equivalence(SGate(), def_s)

# SdgGate

def_sdg = QuantumCircuit(1)
def_sdg.append(U1Gate(-pi / 2), [0], [])
_sel.add_equivalence(SdgGate(), def_sdg)

# SwapGate

def_swap = QuantumCircuit(2)
for inst, qargs, cargs in [(CXGate(), [0, 1], []), (CXGate(), [1, 0], []), (CXGate(), [0, 1], [])]:
    def_swap.append(inst, qargs, cargs)
_sel.add_equivalence(SwapGate(), def_swap)

# iSwapGate

def_iswap = QuantumCircuit(2)
for inst, qargs, cargs in [
    (SGate(), [0], []),
    (SGate(), [1], []),
    (HGate(), [0], []),
    (CXGate(), [0, 1], []),
    (CXGate(), [1, 0], []),
    (HGate(), [1], []),
]:
    def_iswap.append(inst, qargs, cargs)
_sel.add_equivalence(iSwapGate(), def_iswap)

# SXGate

def_sx = QuantumCircuit(1, global_phase=pi / 4)
for inst, qargs, cargs in [(SdgGate(), [0], []), (HGate(), [0], []), (SdgGate(), [0], [])]:
    def_sx.append(inst, qargs, cargs)
_sel.add_equivalence(SXGate(), def_sx)

sx_to_rx = QuantumCircuit(1, global_phase=pi / 4)
sx_to_rx.rx(pi / 2, 0)
_sel.add_equivalence(SXGate(), sx_to_rx)

# SXdgGate

def_sxdg = QuantumCircuit(1, global_phase=-pi / 4)
for inst, qargs, cargs in [(SGate(), [0], []), (HGate(), [0], []), (SGate(), [0], [])]:
    def_sxdg.append(inst, qargs, cargs)
_sel.add_equivalence(SXdgGate(), def_sxdg)

sxdg_to_rx = QuantumCircuit(1, global_phase=-pi / 4)
sxdg_to_rx.rx(-pi / 2, 0)
_sel.add_equivalence(SXdgGate(), sxdg_to_rx)

# CSXGate

def_csx = QuantumCircuit(2)
for inst, qargs, cargs in [(HGate(), [1], []), (CU1Gate(pi / 2), [0, 1], []), (HGate(), [1], [])]:
    def_csx.append(inst, qargs, cargs)
_sel.add_equivalence(CSXGate(), def_csx)

# DCXGate

def_dcx = QuantumCircuit(2)
for inst, qargs, cargs in [(CXGate(), [0, 1], []), (CXGate(), [1, 0], [])]:
    def_dcx.append(inst, qargs, cargs)
_sel.add_equivalence(DCXGate(), def_dcx)

dcx_to_iswap = QuantumCircuit(2)
for inst, qargs, cargs in [
    (HGate(), [0], []),
    (SdgGate(), [0], []),
    (SdgGate(), [1], []),
    (iSwapGate(), [0, 1], []),
    (HGate(), [1], []),
]:
    dcx_to_iswap.append(inst, qargs, cargs)
_sel.add_equivalence(DCXGate(), dcx_to_iswap)

# CSwapGate

def_cswap = QuantumCircuit(3)
for inst, qargs, cargs in [
    (CXGate(), [2, 1], []),
    (CCXGate(), [0, 1, 2], []),
    (CXGate(), [2, 1], []),
]:
    def_cswap.append(inst, qargs, cargs)
_sel.add_equivalence(CSwapGate(), def_cswap)

# TGate

def_t = QuantumCircuit(1)
def_t.append(U1Gate(pi / 4), [0], [])
_sel.add_equivalence(TGate(), def_t)

# TdgGate

def_tdg = QuantumCircuit(1)
def_tdg.append(U1Gate(-pi / 4), [0], [])
_sel.add_equivalence(TdgGate(), def_tdg)

# UGate

theta = Parameter("theta")
phi = Parameter("phi")
lam = Parameter("lam")
u_to_u3 = QuantumCircuit(1)
u_to_u3.append(U3Gate(theta, phi, lam), [0])
_sel.add_equivalence(UGate(theta, phi, lam), u_to_u3)

# CUGate

theta = Parameter("theta")
phi = Parameter("phi")
lam = Parameter("lam")
gamma = Parameter("gamma")
def_cu = QuantumCircuit(2)
def_cu.p(gamma, 0)
def_cu.p((lam + phi) / 2, 0)
def_cu.p((lam - phi) / 2, 1)
def_cu.cx(0, 1)
def_cu.u(-theta / 2, 0, -(phi + lam) / 2, 1)
def_cu.cx(0, 1)
def_cu.u(theta / 2, phi, 0, 1)
_sel.add_equivalence(CUGate(theta, phi, lam, gamma), def_cu)

theta = Parameter("theta")
phi = Parameter("phi")
lam = Parameter("lam")
gamma = Parameter("gamma")
cu_to_cu3 = QuantumCircuit(2)
cu_to_cu3.p(gamma, 0)
cu_to_cu3.append(CU3Gate(theta, phi, lam), [0, 1])
_sel.add_equivalence(CUGate(theta, phi, lam, gamma), cu_to_cu3)

# U1Gate

theta = Parameter("theta")
def_u1 = QuantumCircuit(1)
def_u1.append(U3Gate(0, 0, theta), [0], [])
_sel.add_equivalence(U1Gate(theta), def_u1)

theta = Parameter("theta")
u1_to_phase = QuantumCircuit(1)
u1_to_phase.p(theta, 0)
_sel.add_equivalence(U1Gate(theta), u1_to_phase)

# U1Gate

theta = Parameter("theta")
u1_to_rz = QuantumCircuit(1, global_phase=theta / 2)
u1_to_rz.append(RZGate(theta), [0], [])
_sel.add_equivalence(U1Gate(theta), u1_to_rz)

# CU1Gate

theta = Parameter("theta")
def_cu1 = QuantumCircuit(2)
for inst, qargs, cargs in [
    (U1Gate(theta / 2), [0], []),
    (CXGate(), [0, 1], []),
    (U1Gate(-theta / 2), [1], []),
    (CXGate(), [0, 1], []),
    (U1Gate(theta / 2), [1], []),
]:
    def_cu1.append(inst, qargs, cargs)
_sel.add_equivalence(CU1Gate(theta), def_cu1)

# U1Gate

phi = Parameter("phi")
lam = Parameter("lam")
def_u2 = QuantumCircuit(1)
def_u2.append(U3Gate(pi / 2, phi, lam), [0], [])
_sel.add_equivalence(U2Gate(phi, lam), def_u2)

# U2Gate

phi = Parameter("phi")
lam = Parameter("lam")
u2_to_u1sx = QuantumCircuit(1, global_phase=-pi / 4)
u2_to_u1sx.append(U1Gate(lam - pi / 2), [0])
u2_to_u1sx.sx(0)
u2_to_u1sx.append(U1Gate(phi + pi / 2), [0])
_sel.add_equivalence(U2Gate(phi, lam), u2_to_u1sx)

# U3Gate

theta = Parameter("theta")
phi = Parameter("phi")
lam = Parameter("lam")
u3_qasm_def = QuantumCircuit(1, global_phase=(lam + phi - pi) / 2)
u3_qasm_def.rz(lam, 0)
u3_qasm_def.sx(0)
u3_qasm_def.rz(theta + pi, 0)
u3_qasm_def.sx(0)
u3_qasm_def.rz(phi + 3 * pi, 0)
_sel.add_equivalence(U3Gate(theta, phi, lam), u3_qasm_def)

theta = Parameter("theta")
phi = Parameter("phi")
lam = Parameter("lam")
u3_to_u = QuantumCircuit(1)
u3_to_u.u(theta, phi, lam, 0)
_sel.add_equivalence(U3Gate(theta, phi, lam), u3_to_u)

# CU3Gate

theta = Parameter("theta")
phi = Parameter("phi")
lam = Parameter("lam")
def_cu3 = QuantumCircuit(2)
for inst, qargs, cargs in [
    (U1Gate((lam + phi) / 2), [0], []),
    (U1Gate((lam - phi) / 2), [1], []),
    (CXGate(), [0, 1], []),
    (U3Gate(-theta / 2, 0, -(phi + lam) / 2), [1], []),
    (CXGate(), [0, 1], []),
    (U3Gate(theta / 2, phi, 0), [1], []),
]:
    def_cu3.append(inst, qargs, cargs)
_sel.add_equivalence(CU3Gate(theta, phi, lam), def_cu3)

theta = Parameter("theta")
phi = Parameter("phi")
lam = Parameter("lam")
cu3_to_cu = QuantumCircuit(2)
cu3_to_cu.cu(theta, phi, lam, 0, 0, 1)

# XGate

def_x = QuantumCircuit(1)
def_x.append(U3Gate(pi, 0, pi), [0], [])
_sel.add_equivalence(XGate(), def_x)

# CXGate

for plus_ry in [False, True]:
    for plus_rxx in [False, True]:
        cx_to_rxx = cnot_rxx_decompose(plus_ry, plus_rxx)
        _sel.add_equivalence(CXGate(), cx_to_rxx)

cx_to_cz = QuantumCircuit(2)
for inst, qargs, cargs in [(HGate(), [1], []), (CZGate(), [0, 1], []), (HGate(), [1], [])]:
    cx_to_cz.append(inst, qargs, cargs)
_sel.add_equivalence(CXGate(), cx_to_cz)

cx_to_iswap = QuantumCircuit(2, global_phase=3 * pi / 4)
for inst, qargs, cargs in [
    (HGate(), [0], []),
    (XGate(), [1], []),
    (HGate(), [1], []),
    (iSwapGate(), [0, 1], []),
    (XGate(), [0], []),
    (XGate(), [1], []),
    (HGate(), [1], []),
    (iSwapGate(), [0, 1], []),
    (HGate(), [0], []),
    (SGate(), [0], []),
    (SGate(), [1], []),
    (XGate(), [1], []),
    (HGate(), [1], []),
]:
    cx_to_iswap.append(inst, qargs, cargs)
_sel.add_equivalence(CXGate(), cx_to_iswap)

cx_to_ecr = QuantumCircuit(2, global_phase=-pi / 4)
for inst, qargs, cargs in [
    (RZGate(-pi / 2), [0], []),
    (RYGate(pi), [0], []),
    (RXGate(pi / 2), [1], []),
    (ECRGate(), [0, 1], []),
]:
    cx_to_ecr.append(inst, qargs, cargs)
_sel.add_equivalence(CXGate(), cx_to_ecr)

# CCXGate

def_ccx = QuantumCircuit(3)
for inst, qargs, cargs in [
    (HGate(), [2], []),
    (CXGate(), [1, 2], []),
    (TdgGate(), [2], []),
    (CXGate(), [0, 2], []),
    (TGate(), [2], []),
    (CXGate(), [1, 2], []),
    (TdgGate(), [2], []),
    (CXGate(), [0, 2], []),
    (TGate(), [1], []),
    (TGate(), [2], []),
    (HGate(), [2], []),
    (CXGate(), [0, 1], []),
    (TGate(), [0], []),
    (TdgGate(), [1], []),
    (CXGate(), [0, 1], []),
]:
    def_ccx.append(inst, qargs, cargs)
_sel.add_equivalence(CCXGate(), def_ccx)

# YGate

def_y = QuantumCircuit(1)
def_y.append(U3Gate(pi, pi / 2, pi / 2), [0], [])
_sel.add_equivalence(YGate(), def_y)

# CYGate

def_cy = QuantumCircuit(2)
for inst, qargs, cargs in [(SdgGate(), [1], []), (CXGate(), [0, 1], []), (SGate(), [1], [])]:
    def_cy.append(inst, qargs, cargs)
_sel.add_equivalence(CYGate(), def_cy)

# ZGate

def_z = QuantumCircuit(1)
def_z.append(U1Gate(pi), [0], [])
_sel.add_equivalence(ZGate(), def_z)

# CZGate

def_cz = QuantumCircuit(2)
for inst, qargs, cargs in [(HGate(), [1], []), (CXGate(), [0, 1], []), (HGate(), [1], [])]:
    def_cz.append(inst, qargs, cargs)
_sel.add_equivalence(CZGate(), def_cz)

# RXGate, XGate equivalence

x_to_rx = QuantumCircuit(1)
x_to_rx.append(RXGate(theta=pi), [0])
x_to_rx.global_phase = pi / 2
_sel.add_equivalence(XGate(), x_to_rx)

# RYGate, YGate equivalence

y_to_ry = QuantumCircuit(1)
y_to_ry.append(RYGate(theta=pi), [0])
y_to_ry.global_phase = pi / 2
_sel.add_equivalence(YGate(), y_to_ry)


# HGate, RXGate(pi).RYGate(pi/2) equivalence

h_to_rxry = QuantumCircuit(1)
h_to_rxry.append(RYGate(theta=pi / 2), [0])
h_to_rxry.append(RXGate(theta=pi), [0])
h_to_rxry.global_phase = pi / 2
_sel.add_equivalence(HGate(), h_to_rxry)

# HGate, RGate(pi, 0).RGate(pi/2, pi/2) equivalence

h_to_rr = QuantumCircuit(1)
h_to_rr.append(RGate(theta=pi / 2, phi=pi / 2), [0])
h_to_rr.append(RGate(theta=pi, phi=0), [0])
h_to_rr.global_phase = pi / 2
_sel.add_equivalence(HGate(), h_to_rr)
