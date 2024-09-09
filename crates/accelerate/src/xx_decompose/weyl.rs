use std::f64::consts::PI;
use num_complex::Complex64;
use qiskit_circuit::operations::StandardGate;
use crate::xx_decompose::types::{Coordinate, GateData};
use qiskit_circuit::util::{C_ONE, C_M_ONE, M_IM, IM};

// These names aren't very functional. I think they were in the original
// Python source for debugging purposes. But they were part of the data
// structure. We preserve them for now.
#[derive(Clone, Copy)]
pub(crate) enum ReflectionName {
    NoReflection = 0,
    ReflectXXYY = 1,
    ReflectXXZZ = 2,
    ReflectYYZZ = 3,
}

pub(crate) static REFLECTION_NAMES: [ReflectionName; 4] = [ReflectionName::NoReflection, ReflectionName::ReflectXXYY,
                           ReflectionName::ReflectXXZZ, ReflectionName::ReflectYYZZ,];


// A table of available reflection transformations on canonical coordinates.
// Entries take the form
//     readable_name: (reflection scalars, global phase, [gate constructors]),
// where reflection scalars (a, b, c) model the map (x, y, z) |-> (ax, by, cz),
// global phase is a complex unit, and gate constructors are applied in sequence
// and by conjugation to the first qubit and are passed pi as a parameter.
static reflection_options: [(&[i32; 3], Complex64, &[StandardGate]); 4]  =
    [(&[1, 1, 1], C_ONE, &[]), // 0
     (&[-1, -1, 1], C_ONE, &[StandardGate::RZGate]), // 1
     (&[-1, 1, -1], C_ONE, &[StandardGate::RYGate]), // 2
     (&[1, -1, -1], C_ONE, &[StandardGate::RXGate]), // 3
    ];


#[derive(Clone, Copy)]
pub(crate) enum ShiftName {
    NoShift = 0,
    ZShift = 1,
    YShift = 2,
    YZShift = 3,
    XShift = 4,
    XZShift = 5,
    YYShift = 6,
    XYZShift = 7,
}

pub(crate) static SHIFT_NAMES: [ShiftName; 8] = [
    ShiftName::NoShift,
    ShiftName::ZShift,
    ShiftName::YShift,
    ShiftName::YZShift,
    ShiftName::XShift,
    ShiftName::XZShift,
    ShiftName::YYShift,
    ShiftName::XYZShift,
    ];

static shift_options: [(&[i32; 3], Complex64, &[StandardGate]); 8] =
[(&[0, 0, 0], C_ONE, &[]),
 (&[0, 0, 1], IM, &[StandardGate::RZGate]),
 (&[0, 1, 0], M_IM, &[StandardGate::RYGate]),
 (&[0, 1, 1], C_ONE, &[StandardGate::RYGate, StandardGate::RZGate]),
 (&[1, 0, 0], M_IM, &[StandardGate::RXGate]),
 (&[1, 0, 1], C_ONE, &[StandardGate::RXGate, StandardGate::RZGate]),
 (&[1, 1, 0], C_M_ONE, &[StandardGate::RXGate, StandardGate::RYGate]),
 (&[1, 1, 1], M_IM, &[StandardGate::RXGate, StandardGate::RYGate, StandardGate::RZGate]),
 ];


pub(crate) fn apply_reflection(reflection_name: ReflectionName, coordinate: &Coordinate) ->
    (Coordinate, Vec<GateData>, Complex64)
{
    let (reflection_scalars, reflection_phase_shift, source_reflection_gates) =
        reflection_options[reflection_name as usize];

    let reflected_coord = coordinate.reflect(reflection_scalars);
    let source_reflection: Vec<_> = source_reflection_gates
        .iter()
        .map(|g| GateData::with_param(*g, PI,  0))
        .collect();
    return (reflected_coord, source_reflection, reflection_phase_shift)
}

pub(crate) fn apply_shift(shift_name: ShiftName, coordinate: &Coordinate) ->
    (Coordinate, Vec<GateData>, Complex64)
{
    let (shift_scalars, shift_phase_shift, source_shift_gates) =
        shift_options[shift_name as usize];
    let shifted_coord = coordinate.shift(shift_scalars);

    let source_shift: Vec<_> = source_shift_gates
        .iter()
        .flat_map(|g| [GateData::with_param(*g, PI,  0),
                       GateData::with_param(*g, PI,  1),
        ])
        .collect();
    return (shifted_coord, source_shift, shift_phase_shift)
}


/// Given a pair of distinct indices 0 ≤ (first_index, second_index) ≤ 2,
/// produces a two-qubit circuit which rotates a canonical gate
///
///    a0 XX + a1 YY + a2 ZZ
///
/// into
///
///     a[first] XX + a[second] YY + a[other] ZZ .
pub(crate) fn canonical_rotation_circuit(first_index: usize, second_index: usize) -> Option<Vec<GateData>> {
    let pi2 = Some(PI / 2.0);
    let mpi2 = Some(-PI / 2.0);
    let circuit = match (first_index, second_index) {
        (0, 1) => return None, // Nothing to do.
        (0, 2) =>
            vec![GateData {gate: StandardGate::RXGate, param: mpi2, qubit: 0},
                 GateData {gate: StandardGate::RXGate, param: pi2, qubit: 1},],
        (1, 0) =>
            vec![GateData {gate: StandardGate::RZGate, param: mpi2, qubit: 0},
                 GateData {gate: StandardGate::RZGate, param: pi2, qubit: 1},],
        (1, 2) =>
            vec![GateData {gate: StandardGate::RZGate, param: pi2, qubit: 0},
                 GateData {gate: StandardGate::RZGate, param: pi2, qubit: 1},
                 GateData {gate: StandardGate::RYGate, param: pi2, qubit: 0},
                 GateData {gate: StandardGate::RYGate, param: mpi2, qubit: 1},],
        (2, 0) =>
            vec![GateData {gate: StandardGate::RZGate, param: pi2, qubit: 0},
                 GateData {gate: StandardGate::RZGate, param: pi2, qubit: 1},
                 GateData {gate: StandardGate::RXGate, param: pi2, qubit: 0},
                 GateData {gate: StandardGate::RXGate, param: mpi2, qubit: 1},],
        (2, 1) =>
            vec![GateData {gate: StandardGate::RYGate, param: pi2, qubit: 0},
                 GateData {gate: StandardGate::RYGate, param: mpi2, qubit: 1},],
        (_, _) => unreachable!()
    };
    Some(circuit)
}
