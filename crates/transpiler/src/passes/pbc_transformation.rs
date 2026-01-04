// This code is part of Qiskit.
//
// (C) Copyright IBM 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, FRAC_PI_8, PI};

use qiskit_circuit::operations::StandardGate;

/// Map gates to a list of equivalent Pauli rotations and a global phase.
/// Each element of the list is of the form ((Pauli string, phase rescale factor, [qubit indices]), global phase).
/// For gates that didn't have a phase (e.g. X)
/// the phase rescale factor is simply the phase of the rotation gate. The convention is
/// `original_gate = PauliEvolutionGate(pauli, phase) * e^{i global_phase * phase}`

fn replace_gate_by_pauli_rotation(
    gate: &StandardGate,
) -> (&'static [(&str, f64, &'static [u32])], f64) {
    match gate {
        StandardGate::I => (&[("I", 0.0, &[0])], 0.0),
        StandardGate::X => (&[("X", FRAC_PI_2, &[0])], FRAC_PI_2),
        StandardGate::Y => (&[("Y", FRAC_PI_2, &[0])], FRAC_PI_2),
        StandardGate::Z => (&[("Z", FRAC_PI_2, &[0])], FRAC_PI_2),
        StandardGate::S => (&[("Z", FRAC_PI_4, &[0])], FRAC_PI_4),
        StandardGate::Sdg => (&[("Z", -FRAC_PI_4, &[0])], -FRAC_PI_4),
        StandardGate::T => (&[("Z", FRAC_PI_8, &[0])], FRAC_PI_8),
        StandardGate::Tdg => (&[("Z", -FRAC_PI_8, &[0])], -FRAC_PI_8),
        StandardGate::SX => (&[("X", FRAC_PI_4, &[0])], FRAC_PI_4),
        StandardGate::SXdg => (&[("X", -FRAC_PI_4, &[0])], -FRAC_PI_4),
        StandardGate::H => (
            &[
                ("Z", FRAC_PI_4, &[0]),
                ("X", FRAC_PI_4, &[0]),
                ("Z", FRAC_PI_4, &[0]),
            ],
            FRAC_PI_2,
        ),
        StandardGate::RZ => (&[("Z", 0.5, &[0])], 0.0),
        StandardGate::RX => (&[("X", 0.5, &[0])], 0.0),
        StandardGate::RY => (&[("Y", 0.5, &[0])], 0.0),
        StandardGate::Phase => (&[("Z", 0.5, &[0])], 0.5),
        StandardGate::U1 => (&[("Z", 0.5, &[0])], 0.5),
        StandardGate::CX => (
            &[
                ("XZ", FRAC_PI_4, &[0, 1]),
                ("Z", -FRAC_PI_4, &[0]),
                ("X", -FRAC_PI_4, &[1]),
            ],
            -FRAC_PI_4,
        ),
        StandardGate::CZ => (
            &[
                ("ZZ", FRAC_PI_4, &[0, 1]),
                ("Z", -FRAC_PI_4, &[0]),
                ("Z", -FRAC_PI_4, &[1]),
            ],
            -FRAC_PI_4,
        ),
        StandardGate::CY => (
            &[
                ("YZ", FRAC_PI_4, &[0, 1]),
                ("Z", -FRAC_PI_4, &[0]),
                ("Y", -FRAC_PI_4, &[1]),
            ],
            -FRAC_PI_4,
        ),
        StandardGate::CH => (
            &[
                ("Z", FRAC_PI_2, &[1]),
                ("X", FRAC_PI_4, &[1]),
                ("Z", 3.0 * FRAC_PI_8, &[1]),
                ("XZ", FRAC_PI_4, &[0, 1]),
                ("Z", -FRAC_PI_4, &[0]),
                ("X", -FRAC_PI_4, &[1]),
                ("Z", FRAC_PI_8, &[1]),
                ("X", FRAC_PI_4, &[1]),
            ],
            3.0 * FRAC_PI_4,
        ),
        StandardGate::CS => (
            &[
                ("ZZ", -FRAC_PI_8, &[0, 1]),
                ("Z", FRAC_PI_8, &[0]),
                ("Z", FRAC_PI_8, &[1]),
            ],
            FRAC_PI_8,
        ),
        StandardGate::CSdg => (
            &[
                ("ZZ", FRAC_PI_8, &[0, 1]),
                ("Z", -FRAC_PI_8, &[0]),
                ("Z", -FRAC_PI_8, &[1]),
            ],
            -FRAC_PI_8,
        ),
        StandardGate::CSX => (
            &[
                ("XZ", -FRAC_PI_8, &[0, 1]),
                ("Z", FRAC_PI_8, &[0]),
                ("X", FRAC_PI_8, &[1]),
            ],
            FRAC_PI_8,
        ),
        StandardGate::Swap => (
            &[
                ("XX", FRAC_PI_4, &[0, 1]),
                ("YY", FRAC_PI_4, &[0, 1]),
                ("ZZ", FRAC_PI_4, &[0, 1]),
            ],
            FRAC_PI_4,
        ),
        StandardGate::ISwap => (
            &[("XX", -FRAC_PI_4, &[0, 1]), ("YY", -FRAC_PI_4, &[0, 1])],
            0.0,
        ),
        StandardGate::DCX => (
            &[
                ("XZ", FRAC_PI_4, &[0, 1]),
                ("Z", -FRAC_PI_4, &[0]),
                ("X", -FRAC_PI_4, &[1]),
                ("XZ", FRAC_PI_4, &[0, 1]),
                ("Z", -FRAC_PI_4, &[0]),
                ("X", -FRAC_PI_4, &[1]),
            ],
            -FRAC_PI_2,
        ),
        StandardGate::ECR => (
            &[
                ("Y", -FRAC_PI_2, &[0]),
                ("Z", FRAC_PI_4, &[0]),
                ("X", -FRAC_PI_4, &[1]),
                ("XZ", FRAC_PI_4, &[0, 1]),
                ("Z", -FRAC_PI_4, &[0]),
                ("X", -FRAC_PI_4, &[1]),
            ],
            -2.0 * PI,
        ),
        StandardGate::RZZ => (&[("ZZ", 0.5, &[0, 1])], 0.0),
        StandardGate::RXX => (&[("XX", 0.5, &[0, 1])], 0.0),
        StandardGate::RYY => (&[("YY", 0.5, &[0, 1])], 0.0),
        StandardGate::RZX => (&[("ZX", 0.5, &[0, 1])], 0.0),
        StandardGate::CPhase => (
            &[("ZZ", -0.25, &[0, 1]), ("Z", 0.25, &[0]), ("Z", 0.25, &[1])],
            0.25,
        ),
        StandardGate::CU1 => (
            &[("ZZ", -0.25, &[0, 1]), ("Z", 0.25, &[0]), ("Z", 0.25, &[1])],
            0.25,
        ),
        StandardGate::CRZ => (&[("ZZ", -0.25, &[0, 1]), ("Z", 0.25, &[1])], 0.0),
        StandardGate::CRX => (&[("XZ", -0.25, &[0, 1]), ("X", 0.25, &[1])], 0.0),
        StandardGate::CRY => (&[("YZ", -0.25, &[0, 1]), ("Y", 0.25, &[1])], 0.0),
        _ => unreachable!(
            "This is only called for one and two qubit gates with no paramers or with a single parameter."
        ),
    }
}
