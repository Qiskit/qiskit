// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use num_complex::ComplexFloat;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use qiskit_circuit::BlocksMode;
use qiskit_circuit::Qubit;
use qiskit_circuit::VarsMode;
use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, FRAC_PI_8, PI};

use crate::gate_metrics::rotation_trace_and_dim;
use qiskit_circuit::dag_circuit::{DAGCircuit, NodeType};
use qiskit_circuit::operations::{OperationRef, Param, StandardGate};
use qiskit_circuit::packed_instruction::PackedInstruction;

static ROTATION_GATE_NAMES: [&str; 13] = [
    "rx", "ry", "rz", "p", "u1", "rzz", "rxx", "rzx", "ryy", "cp", "crx", "cry", "crz",
];

type SubstituteSequencePi4<'a> = [(&'a [(StandardGate, &'a [u32])], f64); 16];
type SubstituteSequencePi2<'a> = [(&'a [(StandardGate, &'a [u32])], f64); 8];

const MINIMUM_TOL: f64 = 1e-12;

/// Table for RZ(k * pi / 4) substitutions, with 0 <= k < 15
static RZ_SUBSTITUTIONS: SubstituteSequencePi4 = [
    (&[], 0.0),
    (&[(StandardGate::T, &[0])], -FRAC_PI_8),
    (&[(StandardGate::S, &[0])], -FRAC_PI_4),
    (
        &[(StandardGate::S, &[0]), (StandardGate::T, &[0])],
        -3.0 * FRAC_PI_8,
    ),
    (&[(StandardGate::Z, &[0])], -FRAC_PI_2),
    (
        &[(StandardGate::Z, &[0]), (StandardGate::T, &[0])],
        -5.0 * FRAC_PI_8,
    ),
    (&[(StandardGate::Sdg, &[0])], -3.0 * FRAC_PI_4),
    (&[(StandardGate::Tdg, &[0])], -7.0 * FRAC_PI_8),
    (&[], -PI),
    (&[(StandardGate::T, &[0])], 7.0 * FRAC_PI_8),
    (&[(StandardGate::S, &[0])], 3.0 * FRAC_PI_4),
    (
        &[(StandardGate::S, &[0]), (StandardGate::T, &[0])],
        5.0 * FRAC_PI_8,
    ),
    (&[(StandardGate::Z, &[0])], FRAC_PI_2),
    (
        &[(StandardGate::Z, &[0]), (StandardGate::T, &[0])],
        3.0 * FRAC_PI_8,
    ),
    (&[(StandardGate::Sdg, &[0])], FRAC_PI_4),
    (&[(StandardGate::Tdg, &[0])], FRAC_PI_8),
];

/// Table for RX(k * pi / 4) substitutions, with 0 <= k < 15
static RX_SUBSTITUTIONS: SubstituteSequencePi4 = [
    (&[], 0.0),
    (
        &[
            (StandardGate::H, &[0]),
            (StandardGate::T, &[0]),
            (StandardGate::H, &[0]),
        ],
        -FRAC_PI_8,
    ),
    (&[(StandardGate::SX, &[0])], -FRAC_PI_4),
    (
        &[
            (StandardGate::SX, &[0]),
            (StandardGate::H, &[0]),
            (StandardGate::T, &[0]),
            (StandardGate::H, &[0]),
        ],
        -3.0 * FRAC_PI_8,
    ),
    (&[(StandardGate::X, &[0])], -FRAC_PI_2),
    (
        &[
            (StandardGate::X, &[0]),
            (StandardGate::H, &[0]),
            (StandardGate::T, &[0]),
            (StandardGate::H, &[0]),
        ],
        -5.0 * FRAC_PI_8,
    ),
    (&[(StandardGate::SXdg, &[0])], -3.0 * FRAC_PI_4),
    (
        &[
            (StandardGate::H, &[0]),
            (StandardGate::Tdg, &[0]),
            (StandardGate::H, &[0]),
        ],
        -7.0 * FRAC_PI_8,
    ),
    (&[], -PI),
    (
        &[
            (StandardGate::H, &[0]),
            (StandardGate::T, &[0]),
            (StandardGate::H, &[0]),
        ],
        7.0 * FRAC_PI_8,
    ),
    (&[(StandardGate::SX, &[0])], 3.0 * FRAC_PI_4),
    (
        &[
            (StandardGate::SX, &[0]),
            (StandardGate::H, &[0]),
            (StandardGate::T, &[0]),
            (StandardGate::H, &[0]),
        ],
        5.0 * FRAC_PI_8,
    ),
    (&[(StandardGate::X, &[0])], FRAC_PI_2),
    (
        &[
            (StandardGate::X, &[0]),
            (StandardGate::H, &[0]),
            (StandardGate::T, &[0]),
            (StandardGate::H, &[0]),
        ],
        3.0 * FRAC_PI_8,
    ),
    (&[(StandardGate::SXdg, &[0])], FRAC_PI_4),
    (
        &[
            (StandardGate::H, &[0]),
            (StandardGate::Tdg, &[0]),
            (StandardGate::H, &[0]),
        ],
        FRAC_PI_8,
    ),
];

/// Table for RY(k * pi / 4) substitutions, with 0 <= k < 15
static RY_SUBSTITUTIONS: SubstituteSequencePi4 = [
    (&[], 0.0),
    (
        &[
            (StandardGate::SX, &[0]),
            (StandardGate::T, &[0]),
            (StandardGate::SXdg, &[0]),
        ],
        -FRAC_PI_8,
    ),
    (&[(StandardGate::Z, &[0]), (StandardGate::H, &[0])], 0.0),
    (
        &[
            (StandardGate::SX, &[0]),
            (StandardGate::T, &[0]),
            (StandardGate::S, &[0]),
            (StandardGate::SXdg, &[0]),
        ],
        -3.0 * FRAC_PI_8,
    ),
    (&[(StandardGate::Y, &[0])], -FRAC_PI_2),
    (
        &[
            (StandardGate::Y, &[0]),
            (StandardGate::SX, &[0]),
            (StandardGate::T, &[0]),
            (StandardGate::SXdg, &[0]),
        ],
        -5.0 * FRAC_PI_8,
    ),
    (&[(StandardGate::H, &[0]), (StandardGate::Z, &[0])], -PI),
    (
        &[
            (StandardGate::SX, &[0]),
            (StandardGate::Tdg, &[0]),
            (StandardGate::SXdg, &[0]),
        ],
        -7.0 * FRAC_PI_8,
    ),
    (&[], -PI),
    (
        &[
            (StandardGate::SX, &[0]),
            (StandardGate::T, &[0]),
            (StandardGate::SXdg, &[0]),
        ],
        7.0 * FRAC_PI_8,
    ),
    (&[(StandardGate::Z, &[0]), (StandardGate::H, &[0])], -PI),
    (
        &[
            (StandardGate::SX, &[0]),
            (StandardGate::T, &[0]),
            (StandardGate::S, &[0]),
            (StandardGate::SXdg, &[0]),
        ],
        5.0 * FRAC_PI_8,
    ),
    (&[(StandardGate::Y, &[0])], FRAC_PI_2),
    (
        &[
            (StandardGate::Y, &[0]),
            (StandardGate::SX, &[0]),
            (StandardGate::T, &[0]),
            (StandardGate::SXdg, &[0]),
        ],
        3.0 * FRAC_PI_8,
    ),
    (&[(StandardGate::H, &[0]), (StandardGate::Z, &[0])], 0.0),
    (
        &[
            (StandardGate::SX, &[0]),
            (StandardGate::Tdg, &[0]),
            (StandardGate::SXdg, &[0]),
        ],
        FRAC_PI_8,
    ),
];

/// Table for P(k * pi / 4) substitutions, with 0 <= k < 15
static P_SUBSTITUTIONS: SubstituteSequencePi4 = [
    (&[], 0.0),
    (&[(StandardGate::T, &[0])], 0.0),
    (&[(StandardGate::S, &[0])], 0.0),
    (&[(StandardGate::S, &[0]), (StandardGate::T, &[0])], 0.0),
    (&[(StandardGate::Z, &[0])], 0.0),
    (&[(StandardGate::Z, &[0]), (StandardGate::T, &[0])], 0.0),
    (&[(StandardGate::Sdg, &[0])], 0.0),
    (&[(StandardGate::Tdg, &[0])], 0.0),
    (&[], 0.0),
    (&[(StandardGate::T, &[0])], 0.0),
    (&[(StandardGate::S, &[0])], 0.0),
    (&[(StandardGate::S, &[0]), (StandardGate::T, &[0])], 0.0),
    (&[(StandardGate::Z, &[0])], 0.0),
    (&[(StandardGate::Z, &[0]), (StandardGate::T, &[0])], 0.0),
    (&[(StandardGate::Sdg, &[0])], 0.0),
    (&[(StandardGate::Tdg, &[0])], 0.0),
];

/// Table for RZZ(k * pi / 4) substitutions, with 0 <= k < 15
static RZZ_SUBSTITUTIONS: SubstituteSequencePi4 = [
    (&[], 0.0),
    (
        &[
            (StandardGate::CX, &[0, 1]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
        ],
        -FRAC_PI_8,
    ),
    (
        &[
            (StandardGate::CZ, &[0, 1]),
            (StandardGate::S, &[0]),
            (StandardGate::S, &[1]),
        ],
        -FRAC_PI_4,
    ),
    (
        &[
            (StandardGate::CX, &[0, 1]),
            (StandardGate::S, &[1]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
        ],
        -3.0 * FRAC_PI_8,
    ),
    (
        &[(StandardGate::Z, &[0]), (StandardGate::Z, &[1])],
        -FRAC_PI_2,
    ),
    (
        &[
            (StandardGate::CX, &[0, 1]),
            (StandardGate::Z, &[1]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
        ],
        -5.0 * FRAC_PI_8,
    ),
    (
        &[
            (StandardGate::CZ, &[0, 1]),
            (StandardGate::Sdg, &[0]),
            (StandardGate::Sdg, &[1]),
        ],
        -3.0 * FRAC_PI_4,
    ),
    (
        &[
            (StandardGate::CX, &[0, 1]),
            (StandardGate::Tdg, &[1]),
            (StandardGate::CX, &[0, 1]),
        ],
        -7.0 * FRAC_PI_8,
    ),
    (&[], -PI),
    (
        &[
            (StandardGate::CX, &[0, 1]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
        ],
        7.0 * FRAC_PI_8,
    ),
    (
        &[
            (StandardGate::CZ, &[0, 1]),
            (StandardGate::S, &[0]),
            (StandardGate::S, &[1]),
        ],
        3.0 * FRAC_PI_4,
    ),
    (
        &[
            (StandardGate::CX, &[0, 1]),
            (StandardGate::S, &[1]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
        ],
        5.0 * FRAC_PI_8,
    ),
    (
        &[(StandardGate::Z, &[0]), (StandardGate::Z, &[1])],
        FRAC_PI_2,
    ),
    (
        &[
            (StandardGate::CX, &[0, 1]),
            (StandardGate::Z, &[1]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
        ],
        3.0 * FRAC_PI_8,
    ),
    (
        &[
            (StandardGate::CZ, &[0, 1]),
            (StandardGate::Sdg, &[0]),
            (StandardGate::Sdg, &[1]),
        ],
        FRAC_PI_4,
    ),
    (
        &[
            (StandardGate::CX, &[0, 1]),
            (StandardGate::Tdg, &[1]),
            (StandardGate::CX, &[0, 1]),
        ],
        FRAC_PI_8,
    ),
];

/// Table for RXX(k * pi / 4) substitutions, with 0 <= k < 15
static RXX_SUBSTITUTIONS: SubstituteSequencePi4 = [
    (&[], 0.0),
    (
        &[
            (StandardGate::H, &[0]),
            (StandardGate::H, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::H, &[0]),
            (StandardGate::H, &[1]),
        ],
        -FRAC_PI_8,
    ),
    (
        &[
            (StandardGate::H, &[0]),
            (StandardGate::H, &[1]),
            (StandardGate::CZ, &[0, 1]),
            (StandardGate::S, &[0]),
            (StandardGate::S, &[1]),
            (StandardGate::H, &[0]),
            (StandardGate::H, &[1]),
        ],
        -FRAC_PI_4,
    ),
    (
        &[
            (StandardGate::H, &[0]),
            (StandardGate::H, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::S, &[1]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::H, &[0]),
            (StandardGate::H, &[1]),
        ],
        -3.0 * FRAC_PI_8,
    ),
    (
        &[(StandardGate::X, &[0]), (StandardGate::X, &[1])],
        -FRAC_PI_2,
    ),
    (
        &[
            (StandardGate::H, &[0]),
            (StandardGate::H, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::Z, &[1]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::H, &[0]),
            (StandardGate::H, &[1]),
        ],
        -5.0 * FRAC_PI_8,
    ),
    (
        &[
            (StandardGate::H, &[0]),
            (StandardGate::H, &[1]),
            (StandardGate::CZ, &[0, 1]),
            (StandardGate::Sdg, &[0]),
            (StandardGate::Sdg, &[1]),
            (StandardGate::H, &[0]),
            (StandardGate::H, &[1]),
        ],
        -3.0 * FRAC_PI_4,
    ),
    (
        &[
            (StandardGate::H, &[0]),
            (StandardGate::H, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::Tdg, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::H, &[0]),
            (StandardGate::H, &[1]),
        ],
        -7.0 * FRAC_PI_8,
    ),
    (&[], -PI),
    (
        &[
            (StandardGate::H, &[0]),
            (StandardGate::H, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::H, &[0]),
            (StandardGate::H, &[1]),
        ],
        7.0 * FRAC_PI_8,
    ),
    (
        &[
            (StandardGate::H, &[0]),
            (StandardGate::H, &[1]),
            (StandardGate::CZ, &[0, 1]),
            (StandardGate::S, &[0]),
            (StandardGate::S, &[1]),
            (StandardGate::H, &[0]),
            (StandardGate::H, &[1]),
        ],
        3.0 * FRAC_PI_4,
    ),
    (
        &[
            (StandardGate::H, &[0]),
            (StandardGate::H, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::S, &[1]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::H, &[0]),
            (StandardGate::H, &[1]),
        ],
        5.0 * FRAC_PI_8,
    ),
    (
        &[(StandardGate::X, &[0]), (StandardGate::X, &[1])],
        FRAC_PI_2,
    ),
    (
        &[
            (StandardGate::H, &[0]),
            (StandardGate::H, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::Z, &[1]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::H, &[0]),
            (StandardGate::H, &[1]),
        ],
        3.0 * FRAC_PI_8,
    ),
    (
        &[
            (StandardGate::H, &[0]),
            (StandardGate::H, &[1]),
            (StandardGate::CZ, &[0, 1]),
            (StandardGate::Sdg, &[0]),
            (StandardGate::Sdg, &[1]),
            (StandardGate::H, &[0]),
            (StandardGate::H, &[1]),
        ],
        FRAC_PI_4,
    ),
    (
        &[
            (StandardGate::H, &[0]),
            (StandardGate::H, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::Tdg, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::H, &[0]),
            (StandardGate::H, &[1]),
        ],
        FRAC_PI_8,
    ),
];

/// Table for RZX(k * pi / 4) substitutions, with 0 <= k < 15
static RZX_SUBSTITUTIONS: SubstituteSequencePi4 = [
    (&[], 0.0),
    (
        &[
            (StandardGate::H, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::H, &[1]),
        ],
        -FRAC_PI_8,
    ),
    (
        &[
            (StandardGate::H, &[1]),
            (StandardGate::CZ, &[0, 1]),
            (StandardGate::S, &[0]),
            (StandardGate::S, &[1]),
            (StandardGate::H, &[1]),
        ],
        -FRAC_PI_4,
    ),
    (
        &[
            (StandardGate::H, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::S, &[1]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::H, &[1]),
        ],
        -3.0 * FRAC_PI_8,
    ),
    (
        &[(StandardGate::Z, &[0]), (StandardGate::X, &[1])],
        -FRAC_PI_2,
    ),
    (
        &[
            (StandardGate::H, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::Z, &[1]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::H, &[1]),
        ],
        -5.0 * FRAC_PI_8,
    ),
    (
        &[
            (StandardGate::H, &[1]),
            (StandardGate::CZ, &[0, 1]),
            (StandardGate::Sdg, &[0]),
            (StandardGate::Sdg, &[1]),
            (StandardGate::H, &[1]),
        ],
        -3.0 * FRAC_PI_4,
    ),
    (
        &[
            (StandardGate::H, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::Tdg, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::H, &[1]),
        ],
        -7.0 * FRAC_PI_8,
    ),
    (&[], -PI),
    (
        &[
            (StandardGate::H, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::H, &[1]),
        ],
        7.0 * FRAC_PI_8,
    ),
    (
        &[
            (StandardGate::H, &[1]),
            (StandardGate::CZ, &[0, 1]),
            (StandardGate::S, &[0]),
            (StandardGate::S, &[1]),
            (StandardGate::H, &[1]),
        ],
        3.0 * FRAC_PI_4,
    ),
    (
        &[
            (StandardGate::H, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::S, &[1]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::H, &[1]),
        ],
        5.0 * FRAC_PI_8,
    ),
    (
        &[(StandardGate::Z, &[0]), (StandardGate::X, &[1])],
        FRAC_PI_2,
    ),
    (
        &[
            (StandardGate::H, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::Z, &[1]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::H, &[1]),
        ],
        3.0 * FRAC_PI_8,
    ),
    (
        &[
            (StandardGate::H, &[1]),
            (StandardGate::CZ, &[0, 1]),
            (StandardGate::Sdg, &[0]),
            (StandardGate::Sdg, &[1]),
            (StandardGate::H, &[1]),
        ],
        FRAC_PI_4,
    ),
    (
        &[
            (StandardGate::H, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::Tdg, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::H, &[1]),
        ],
        FRAC_PI_8,
    ),
];

/// Table for RYY(k * pi / 4) substitutions, with 0 <= k < 15
static RYY_SUBSTITUTIONS: SubstituteSequencePi4 = [
    (&[], 0.0),
    (
        &[
            (StandardGate::SXdg, &[0]),
            (StandardGate::SXdg, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::SX, &[0]),
            (StandardGate::SX, &[1]),
        ],
        -FRAC_PI_8,
    ),
    (
        &[
            (StandardGate::SXdg, &[0]),
            (StandardGate::SXdg, &[1]),
            (StandardGate::CZ, &[0, 1]),
            (StandardGate::S, &[0]),
            (StandardGate::S, &[1]),
            (StandardGate::SX, &[0]),
            (StandardGate::SX, &[1]),
        ],
        -FRAC_PI_4,
    ),
    (
        &[
            (StandardGate::SXdg, &[0]),
            (StandardGate::SXdg, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::S, &[1]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::SX, &[0]),
            (StandardGate::SX, &[1]),
        ],
        -3.0 * FRAC_PI_8,
    ),
    (
        &[
            (StandardGate::Sdg, &[0]),
            (StandardGate::Sdg, &[1]),
            (StandardGate::X, &[0]),
            (StandardGate::X, &[1]),
            (StandardGate::S, &[0]),
            (StandardGate::S, &[1]),
        ],
        -FRAC_PI_2,
    ),
    (
        &[
            (StandardGate::SXdg, &[0]),
            (StandardGate::SXdg, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::Z, &[1]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::SX, &[0]),
            (StandardGate::SX, &[1]),
        ],
        -5.0 * FRAC_PI_8,
    ),
    (
        &[
            (StandardGate::SXdg, &[0]),
            (StandardGate::SXdg, &[1]),
            (StandardGate::CZ, &[0, 1]),
            (StandardGate::Sdg, &[0]),
            (StandardGate::Sdg, &[1]),
            (StandardGate::SX, &[0]),
            (StandardGate::SX, &[1]),
        ],
        -3.0 * FRAC_PI_4,
    ),
    (
        &[
            (StandardGate::SXdg, &[0]),
            (StandardGate::SXdg, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::Tdg, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::SX, &[0]),
            (StandardGate::SX, &[1]),
        ],
        -7.0 * FRAC_PI_8,
    ),
    (&[], -PI),
    (
        &[
            (StandardGate::SXdg, &[0]),
            (StandardGate::SXdg, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::SX, &[0]),
            (StandardGate::SX, &[1]),
        ],
        7.0 * FRAC_PI_8,
    ),
    (
        &[
            (StandardGate::SXdg, &[0]),
            (StandardGate::SXdg, &[1]),
            (StandardGate::CZ, &[0, 1]),
            (StandardGate::S, &[0]),
            (StandardGate::S, &[1]),
            (StandardGate::SX, &[0]),
            (StandardGate::SX, &[1]),
        ],
        3.0 * FRAC_PI_4,
    ),
    (
        &[
            (StandardGate::SXdg, &[0]),
            (StandardGate::SXdg, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::S, &[1]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::SX, &[0]),
            (StandardGate::SX, &[1]),
        ],
        5.0 * FRAC_PI_8,
    ),
    (
        &[
            (StandardGate::Sdg, &[0]),
            (StandardGate::Sdg, &[1]),
            (StandardGate::X, &[0]),
            (StandardGate::X, &[1]),
            (StandardGate::S, &[0]),
            (StandardGate::S, &[1]),
        ],
        FRAC_PI_2,
    ),
    (
        &[
            (StandardGate::SXdg, &[0]),
            (StandardGate::SXdg, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::Z, &[1]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::SX, &[0]),
            (StandardGate::SX, &[1]),
        ],
        3.0 * FRAC_PI_8,
    ),
    (
        &[
            (StandardGate::SXdg, &[0]),
            (StandardGate::SXdg, &[1]),
            (StandardGate::CZ, &[0, 1]),
            (StandardGate::Sdg, &[0]),
            (StandardGate::Sdg, &[1]),
            (StandardGate::SX, &[0]),
            (StandardGate::SX, &[1]),
        ],
        FRAC_PI_4,
    ),
    (
        &[
            (StandardGate::SXdg, &[0]),
            (StandardGate::SXdg, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::Tdg, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::SX, &[0]),
            (StandardGate::SX, &[1]),
        ],
        FRAC_PI_8,
    ),
];

/// Table for CPhase(k * pi / 2) substitutions, with 0 <= k < 7
static CP_SUBSTITUTIONS: SubstituteSequencePi2 = [
    (&[], 0.0),
    (
        &[
            (StandardGate::T, &[0]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::Tdg, &[1]),
            (StandardGate::CX, &[0, 1]),
        ],
        0.0,
    ),
    (&[(StandardGate::CZ, &[0, 1])], 0.0),
    (
        &[
            (StandardGate::Tdg, &[0]),
            (StandardGate::Tdg, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
        ],
        0.0,
    ),
    (&[], 0.0),
    (
        &[
            (StandardGate::T, &[0]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::Tdg, &[1]),
            (StandardGate::CX, &[0, 1]),
        ],
        0.0,
    ),
    (&[(StandardGate::CZ, &[0, 1])], 0.0),
    (
        &[
            (StandardGate::Tdg, &[0]),
            (StandardGate::Tdg, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
        ],
        0.0,
    ),
];

/// Table for CRZ(k * pi / 2) substitutions, with 0 <= k < 7
static CRZ_SUBSTITUTIONS: SubstituteSequencePi2 = [
    (&[], 0.0),
    (
        &[
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::Tdg, &[1]),
            (StandardGate::CX, &[0, 1]),
        ],
        0.0,
    ),
    (
        &[(StandardGate::CZ, &[0, 1]), (StandardGate::Sdg, &[0])],
        0.0,
    ),
    (
        &[
            (StandardGate::T, &[1]),
            (StandardGate::S, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::Tdg, &[1]),
            (StandardGate::Sdg, &[1]),
            (StandardGate::CX, &[0, 1]),
        ],
        0.0,
    ),
    (&[(StandardGate::Z, &[0])], 0.0),
    (
        &[
            (StandardGate::Tdg, &[1]),
            (StandardGate::Sdg, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::T, &[1]),
            (StandardGate::S, &[1]),
            (StandardGate::CX, &[0, 1]),
        ],
        0.0,
    ),
    (&[(StandardGate::CZ, &[0, 1]), (StandardGate::S, &[0])], 0.0),
    (
        &[
            (StandardGate::Tdg, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
        ],
        0.0,
    ),
];

/// Table for CRX(k * pi / 2) substitutions, with 0 <= k < 7
static CRX_SUBSTITUTIONS: SubstituteSequencePi2 = [
    (&[], 0.0),
    (
        &[
            (StandardGate::H, &[1]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::Tdg, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::H, &[1]),
        ],
        0.0,
    ),
    (
        &[(StandardGate::CX, &[0, 1]), (StandardGate::Sdg, &[0])],
        0.0,
    ),
    (
        &[
            (StandardGate::H, &[1]),
            (StandardGate::T, &[1]),
            (StandardGate::S, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::Tdg, &[1]),
            (StandardGate::Sdg, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::H, &[1]),
        ],
        0.0,
    ),
    (&[(StandardGate::Z, &[0])], 0.0),
    (
        &[
            (StandardGate::H, &[1]),
            (StandardGate::Tdg, &[1]),
            (StandardGate::Sdg, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::T, &[1]),
            (StandardGate::S, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::H, &[1]),
        ],
        0.0,
    ),
    (&[(StandardGate::CX, &[0, 1]), (StandardGate::S, &[0])], 0.0),
    (
        &[
            (StandardGate::H, &[1]),
            (StandardGate::Tdg, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::H, &[1]),
        ],
        0.0,
    ),
];

/// Table for CRY(k * pi / 2) substitutions, with 0 <= k < 7
static CRY_SUBSTITUTIONS: SubstituteSequencePi2 = [
    (&[], 0.0),
    (
        &[
            (StandardGate::SX, &[1]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::Tdg, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::SXdg, &[1]),
        ],
        0.0,
    ),
    (
        &[
            (StandardGate::SX, &[1]),
            (StandardGate::CZ, &[0, 1]),
            (StandardGate::Sdg, &[0]),
            (StandardGate::SXdg, &[1]),
        ],
        0.0,
    ),
    (
        &[
            (StandardGate::SX, &[1]),
            (StandardGate::S, &[1]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::Tdg, &[1]),
            (StandardGate::Sdg, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::SXdg, &[1]),
        ],
        0.0,
    ),
    (&[(StandardGate::Z, &[0])], 0.0),
    (
        &[
            (StandardGate::SX, &[1]),
            (StandardGate::Sdg, &[1]),
            (StandardGate::Tdg, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::T, &[1]),
            (StandardGate::S, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::SXdg, &[1]),
        ],
        0.0,
    ),
    (
        &[
            (StandardGate::SX, &[1]),
            (StandardGate::CZ, &[0, 1]),
            (StandardGate::S, &[0]),
            (StandardGate::SXdg, &[1]),
        ],
        0.0,
    ),
    (
        &[
            (StandardGate::SX, &[1]),
            (StandardGate::Tdg, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::SXdg, &[1]),
        ],
        0.0,
    ),
];

/// For a given angle, if it is a multiple of PI/k, calculate the multiple mod (4*k),
/// Otherwise, return `None`.
fn is_angle_close_to_multiple_of_pi_k(
    gate: StandardGate,
    k: usize,
    angle: f64,
    tol: f64,
) -> Option<usize> {
    let closest_ratio = angle * (k as f64) / PI;
    let closest_integer = closest_ratio.round();
    let closest_angle = closest_integer * PI / (k as f64);
    let theta = angle - closest_angle;

    // Trace and dimension calculation of rotation matrices
    let (tr_over_dim, dim) = rotation_trace_and_dim(gate, theta)
        .expect("Since only supported rotation gates are given, the result is not None");

    // fidelity-based tolerance
    let f_pro = tr_over_dim.abs().powi(2);
    let gate_fidelity = (dim * f_pro + 1.) / (dim + 1.);
    let rem = (4 * k).try_into().unwrap();
    if (1. - gate_fidelity).abs() < tol {
        Some((closest_integer as i64).rem_euclid(rem) as usize)
    } else {
        None
    }
}

/// Gets a rotation gate and outputs an equivalent vector of standard gates
/// in {Clifford, T, Tdg} and a global phase, when the angle is a multiple of pi/4.
/// Note that odd multiples of pi/4 require a single T or Tdg gate
/// as well as some Clifford gates,
/// while even multiples of pi/4, or equivalently, integer multiples of pi/2,
/// can be written using only Clifford gates.
/// The output contains at most one T or Tdg gate, and an optimal number of
/// Clifford gates.
fn replace_rotation_by_discrete(
    gate: StandardGate,
    multiple: usize,
) -> (&'static [(StandardGate, &'static [u32])], f64) {
    match gate {
        StandardGate::RX => RX_SUBSTITUTIONS[multiple],
        StandardGate::RY => RY_SUBSTITUTIONS[multiple],
        StandardGate::RZ => RZ_SUBSTITUTIONS[multiple],
        StandardGate::Phase => P_SUBSTITUTIONS[multiple],
        StandardGate::U1 => P_SUBSTITUTIONS[multiple],
        StandardGate::RZZ => RZZ_SUBSTITUTIONS[multiple],
        StandardGate::RXX => RXX_SUBSTITUTIONS[multiple],
        StandardGate::RZX => RZX_SUBSTITUTIONS[multiple],
        StandardGate::RYY => RYY_SUBSTITUTIONS[multiple],
        StandardGate::CPhase => CP_SUBSTITUTIONS[multiple],
        StandardGate::CRZ => CRZ_SUBSTITUTIONS[multiple],
        StandardGate::CRX => CRX_SUBSTITUTIONS[multiple],
        StandardGate::CRY => CRY_SUBSTITUTIONS[multiple],
        _ => unreachable!("This is only called for rotation gates."),
    }
}

/// Matches 4 to the single-qubit and two-qubit rotation gates, and 2 to the controlled-rotation gates
fn rotation_to_pi_div(gate: StandardGate) -> usize {
    match gate {
        StandardGate::RX => 4,
        StandardGate::RY => 4,
        StandardGate::RZ => 4,
        StandardGate::Phase => 4,
        StandardGate::U1 => 4,
        StandardGate::RZZ => 4,
        StandardGate::RXX => 4,
        StandardGate::RZX => 4,
        StandardGate::RYY => 4,
        StandardGate::CPhase => 2,
        StandardGate::CRZ => 2,
        StandardGate::CRX => 2,
        StandardGate::CRY => 2,
        _ => unreachable!("This is only called for rotation gates."),
    }
}

#[pyfunction]
#[pyo3(name = "substitute_pi4_rotations")]
pub fn py_run_substitute_pi4_rotations(
    dag: &DAGCircuit,
    approximation_degree: f64,
) -> PyResult<Option<DAGCircuit>> {
    // Skip the pass if there are no rotation gates.
    if dag
        .get_op_counts()
        .keys()
        .all(|k| !ROTATION_GATE_NAMES.contains(&k.as_str()))
    {
        return Ok(None);
    }

    let mut new_dag = dag.copy_empty_like_with_capacity(0, 0, VarsMode::Alike, BlocksMode::Keep)?;

    // Iterate over nodes in the DAG and collect nodes that are rotation gates
    // with an angle that is sufficiently close to a multiple of pi/4
    let tol = MINIMUM_TOL.max(1.0 - approximation_degree);
    let mut global_phase_update: f64 = 0.;

    for node_index in dag.topological_op_nodes(false)? {
        if let NodeType::Operation(inst) = &dag[node_index] {
            if let OperationRef::StandardGate(gate) = inst.op.view() {
                if matches!(
                    gate,
                    StandardGate::RX
                        | StandardGate::RY
                        | StandardGate::RZ
                        | StandardGate::Phase
                        | StandardGate::U1
                        | StandardGate::RZZ
                        | StandardGate::RXX
                        | StandardGate::RZX
                        | StandardGate::RYY
                        | StandardGate::CPhase
                        | StandardGate::CRZ
                        | StandardGate::CRX
                        | StandardGate::CRY
                ) {
                    let k = rotation_to_pi_div(gate);
                    if let Param::Float(angle) = inst.params_view()[0] {
                        if let Some(multiple) =
                            is_angle_close_to_multiple_of_pi_k(gate, k, angle, tol)
                        {
                            let (sequence, phase_update) =
                                replace_rotation_by_discrete(gate, multiple);
                            for (new_gate, qubits) in sequence {
                                let original_qubits = dag.get_qargs(inst.qubits);
                                let updated_qubits: Vec<Qubit> = qubits
                                    .iter()
                                    .map(|q| original_qubits[*q as usize])
                                    .collect();
                                let new_qubits = new_dag.add_qargs(&updated_qubits);
                                new_dag.push_back(PackedInstruction::from_standard_gate(
                                    *new_gate, None, new_qubits,
                                ))?;
                            }
                            global_phase_update += phase_update;
                        } else {
                            new_dag.push_back(inst.clone())?;
                        }
                    } else {
                        new_dag.push_back(inst.clone())?;
                    }
                } else {
                    new_dag.push_back(inst.clone())?;
                }
            } else {
                new_dag.push_back(inst.clone())?;
            }
        } else {
            unreachable!();
        }
    }

    new_dag.add_global_phase(&Param::Float(global_phase_update))?;

    Ok(Some(new_dag))
}

pub fn substitute_pi4_rotations_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(py_run_substitute_pi4_rotations))?;
    Ok(())
}
