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

use crate::synthesis::common::SynthesisData;

use std::f64::consts::PI;

// const PI2: f64 = PI / 2.;
// const PI4: f64 = PI / 4.;
const PI8: f64 = PI / 8.;

/// Efficient synthesis for 3-controlled X-gate.
pub fn c3x<'a>() -> SynthesisData<'a> {
    let mut qc = SynthesisData::new(4);
    qc.h(3);
    qc.p(PI8, 0);
    qc.p(PI8, 1);
    qc.p(PI8, 2);
    qc.p(PI8, 3);
    qc.cx(0, 1);
    qc.p(-PI8, 1);
    qc.cx(0, 1);
    qc.cx(1, 2);
    qc.p(-PI8, 2);
    qc.cx(0, 2);
    qc.p(PI8, 2);
    qc.cx(1, 2);
    qc.p(-PI8, 2);
    qc.cx(0, 2);
    qc.cx(2, 3);
    qc.p(-PI8, 3);
    qc.cx(1, 3);
    qc.p(PI8, 3);
    qc.cx(2, 3);
    qc.p(-PI8, 3);
    qc.cx(0, 3);
    qc.p(PI8, 3);
    qc.cx(2, 3);
    qc.p(-PI8, 3);
    qc.cx(1, 3);
    qc.p(PI8, 3);
    qc.cx(2, 3);
    qc.p(-PI8, 3);
    qc.cx(0, 3);
    qc.h(3);
    qc
}
