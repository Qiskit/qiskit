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

use qiskit_circuit::operations::StandardGate;

use std::f64::consts::PI;

const PI4: f64 = PI / 4.0;
const PI8: f64 = PI / 8.0;
const DEFAULT_ATOL: f64 = 1e-10;

/// For an angle, if it is a multiple of 2*PI/8, calculate the multiplicity mod 8,
/// Otherwise, return None.
fn is_angle_close_to_multiple_of_2pi_8(angle: f64) -> Option<usize> {
    let closest_ratio = angle * 4.0 / PI;
    let closest_integer = closest_ratio.round();
    if (closest_ratio - closest_integer).abs() < DEFAULT_ATOL {
        Some((closest_integer as usize + 8) % 8)
    } else {
        None
    }
}

/// Gets a rotation gate (RX/RY/RZ) and outputs an equivalent vector of standard gates and
/// a global phase, when the gate is sufficiently close to Clifford+T/Tdg.
/// Otherwise, return the original gate and global phase = 0.
fn try_replace_rotation_by_discrete(gate: StandardGate, angle: f64) -> (Vec<StandardGate>, f64) {
    let multiple = is_angle_close_to_multiple_of_2pi_8(angle);
    let mut discrete_sequence = Vec::<StandardGate>::with_capacity(4);
    let mut global_phase = 0.0;

    match (gate, multiple) {
        (StandardGate::RZ, Some(0)) => {
            discrete_sequence.push(StandardGate::I);
            global_phase = 0.0;
        }
        (StandardGate::RZ, Some(1)) => {
            discrete_sequence.push(StandardGate::T);
            global_phase = -1.0 * PI8;
        }
        (StandardGate::RZ, Some(2)) => {
            discrete_sequence.push(StandardGate::S);
            global_phase = -2.0 * PI8;
        }
        (StandardGate::RZ, Some(3)) => {
            discrete_sequence.push(StandardGate::S);
            discrete_sequence.push(StandardGate::T);
            global_phase = -3.0 * PI8;
        }
        (StandardGate::RZ, Some(4)) => {
            discrete_sequence.push(StandardGate::Z);
            global_phase = -4.0 * PI8;
        }
        (StandardGate::RZ, Some(5)) => {
            discrete_sequence.push(StandardGate::Z);
            discrete_sequence.push(StandardGate::T);
            global_phase = -5.0 * PI8;
        }
        (StandardGate::RZ, Some(6)) => {
            discrete_sequence.push(StandardGate::Sdg);
            global_phase = -6.0 * PI8;
        }
        (StandardGate::RZ, Some(7)) => {
            discrete_sequence.push(StandardGate::Tdg);
            global_phase = -7.0 * PI8;
        }
        (StandardGate::RX, Some(0)) => {
            discrete_sequence.push(StandardGate::I);
            global_phase = 0.0;
        }
        (StandardGate::RX, Some(1)) => {
            discrete_sequence.push(StandardGate::H);
            discrete_sequence.push(StandardGate::T);
            discrete_sequence.push(StandardGate::H);
            global_phase = -1.0 * PI8;
        }
        (StandardGate::RX, Some(2)) => {
            discrete_sequence.push(StandardGate::SX);
            global_phase = -2.0 * PI8;
        }
        (StandardGate::RX, Some(3)) => {
            discrete_sequence.push(StandardGate::SX);
            discrete_sequence.push(StandardGate::H);
            discrete_sequence.push(StandardGate::T);
            discrete_sequence.push(StandardGate::H);
            global_phase = -3.0 * PI8;
        }
        (StandardGate::RX, Some(4)) => {
            discrete_sequence.push(StandardGate::X);
            global_phase = -4.0 * PI8;
        }
        (StandardGate::RX, Some(5)) => {
            discrete_sequence.push(StandardGate::X);
            discrete_sequence.push(StandardGate::H);
            discrete_sequence.push(StandardGate::T);
            discrete_sequence.push(StandardGate::H);
            global_phase = -5.0 * PI8;
        }
        (StandardGate::RX, Some(6)) => {
            discrete_sequence.push(StandardGate::SXdg);
            global_phase = -6.0 * PI8;
        }
        (StandardGate::RX, Some(7)) => {
            discrete_sequence.push(StandardGate::H);
            discrete_sequence.push(StandardGate::Tdg);
            discrete_sequence.push(StandardGate::H);
            global_phase = -7.0 * PI8;
        }
        (StandardGate::RY, Some(0)) => {
            discrete_sequence.push(StandardGate::I);
            global_phase = 0.0;
        }
        (StandardGate::RY, Some(1)) => {
            discrete_sequence.push(StandardGate::SX);
            discrete_sequence.push(StandardGate::T);
            discrete_sequence.push(StandardGate::SXdg);
            global_phase = -1.0 * PI8;
        }
        (StandardGate::RY, Some(2)) => {
            discrete_sequence.push(StandardGate::Z);
            discrete_sequence.push(StandardGate::H);
            global_phase = 0.0;
        }
        (StandardGate::RY, Some(3)) => {
            discrete_sequence.push(StandardGate::SX);
            discrete_sequence.push(StandardGate::T);
            discrete_sequence.push(StandardGate::S);
            discrete_sequence.push(StandardGate::SXdg);
            global_phase = -3.0 * PI8;
        }
        (StandardGate::RY, Some(4)) => {
            discrete_sequence.push(StandardGate::Y);
            global_phase = -4.0 * PI8;
        }
        (StandardGate::RY, Some(5)) => {
            discrete_sequence.push(StandardGate::Y);
            discrete_sequence.push(StandardGate::SX);
            discrete_sequence.push(StandardGate::T);
            discrete_sequence.push(StandardGate::SXdg);
            global_phase = -5.0 * PI8;
        }
        (StandardGate::RY, Some(6)) => {
            discrete_sequence.push(StandardGate::H);
            discrete_sequence.push(StandardGate::Z);
            global_phase = -1.0 * PI;
        }
        (StandardGate::RY, Some(7)) => {
            discrete_sequence.push(StandardGate::SX);
            discrete_sequence.push(StandardGate::Tdg);
            discrete_sequence.push(StandardGate::SXdg);
            global_phase = -7.0 * PI8;
        }
        _ => discrete_sequence.push(gate),
    }

    (discrete_sequence, global_phase)
}
