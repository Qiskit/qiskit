pub(crate) const EPSILON: f64 = 1e-6;

use std::f64::consts::PI;

// The logic in `safe_acos` is copied from the Python original.
// The following one-line comment is copied from the Python as well.
// TODO: THIS IS A STOPGAP!!!
//
/// Has the same behavior as `f64::acos` except that an
/// argument a bit greater than `1` or less than `-1` is
/// valid and returns the value at `1` or `-1`.
/// Larger or smaller arguments will cause the same error to be
/// raised that `f64::acos` raises.
pub(crate) fn safe_acos(numerator: f64, denominator: f64) -> f64 {
    let threshold: f64 = 0.005;
    if numerator.abs() > denominator.abs() {
        if (numerator - denominator).abs() < threshold {
            return 0.0;
        } else if (numerator + denominator).abs() < threshold {
            return PI;
        }
    }
    (numerator / denominator).acos()
}

// powi(2) everywhere is a bit ugly
pub trait Square {
    fn sq(&self) -> f64;
}

impl Square for f64 {
    fn sq(&self) -> f64 {
        self * self
    }
}
