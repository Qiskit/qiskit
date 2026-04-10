// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use std::f64::consts::PI;

#[derive(Debug, Clone, PartialEq)]
pub enum PIFormat {
    Text,
    Qasm,
    Latex,
    Mpl,
}

const DENOMINATOR: i64 = 16;

/// """Computes if a number is close to an integer
/// fraction or multiple of PI and returns the
/// corresponding string.

/// Args:
///     f (float): Number to check.
///     eps (float): EPS to check against.
///     output (str): Options are 'text' (default),
///                   'latex', 'mpl', and 'qasm'.
///     ndigits (usize or None): Number of digits to print
///                            if returning raw inpt.
///                            If `None` (default), Rust's
///                            default float formatting is used.

/// Returns:
///     Option<String>: string representation of output. None if no pi format is found

pub fn pi_check(
    f: f64,
    eps: Option<f64>,
    output: Option<PIFormat>,
    ndigits: Option<usize>,
) -> Option<String> {
    let output = output.unwrap_or(PIFormat::Text);

    // epsilon value defines the treshold to detect pi.
    let eps: f64 = eps.unwrap_or(1e-9);

    // pi_str is needed to match the output expected according to the format needed
    let pi_str: String = match output {
        PIFormat::Text => "π".to_string(),
        PIFormat::Qasm => "pi".to_string(),
        PIFormat::Latex => "\\pi".to_string(),
        PIFormat::Mpl => "$\\pi$".to_string(),
    };

    // f_abs and sign help us working trough each steps
    let f_abs = f.abs();
    let sign: &str = if f < 0.0 { "-" } else { "" };

    // Detecting 0 before moving on
    if f_abs < eps {
        return Some("0".to_string());
    }

    // First check is for whole multiples of pi
    let val = f_abs / PI;
    if val >= 1.0 - eps {
        let round = val.round();
        if (val - round).abs() < eps {
            let n = round as i64;
            let str_out = match (n, output) {
                (1, _) => format!("{}{}", sign, pi_str),
                (n, PIFormat::Qasm) => format!("{}{}*{}", sign, n, pi_str),
                (n, _) => format!("{}{}{}", sign, n, pi_str),
            };
            return Some(str_out);
        }
    }

    // Second is a check for powers of pi
    if f_abs > PI {
        for k in 2i32..=4 {
            let pow = PI.powi(k);
            if (f_abs - pow).abs() < eps {
                let str_out = match output {
                    PIFormat::Qasm => format_raw(f, ndigits),
                    PIFormat::Latex => format!("{}{}^{}", sign, pi_str, k),
                    PIFormat::Mpl => format!("{}{}^{}$", sign, pi_str, k),
                    PIFormat::Text => format!("{}{}**{}", sign, pi_str, k),
                };
                return Some(str_out);
            }
        }
    }

    // Third is a check for a number larger than MAX_FRAC * pi, not a
    // multiple or power of pi, since no fractions will exceed MAX_FRAC * pi
    if f_abs > (DENOMINATOR as f64 * PI) {
        return Some(format_raw(f, ndigits));
    }

    // Fourth check is for fractions for 1*pi in the numer and any
    // number in the denom.
    let val = PI / f_abs;
    let round = val.round();
    if round >= 1.0 && (val - round).abs() < eps {
        let d = round as i64;
        let str_out = match output {
            PIFormat::Latex => format!("\\frac{{{}{}}}{{{}}}", sign, pi_str, d),
            _ => format!("{}{}/{}", sign, pi_str, d),
        };
        return Some(str_out);
    }

    // Fifth check is for fractions where the numer > 1*pi and numer
    // is up to DENOMINATOR*pi and denom is up to DENOMINATOR and all
    // fractions are reduced. Ex. 15pi/16, 2pi/5, 15pi/2, 16pi/9.
    for d in 1i64..=DENOMINATOR {
        for p in 1i64..=DENOMINATOR {
            let val = (p as f64 / d as f64) * PI;
            if (f_abs - val).abs() < eps {
                let str_out = match output {
                    PIFormat::Latex => format!("\\frac{{{}{}{}}}{{{}}}", sign, p, pi_str, d),
                    PIFormat::Qasm => format!("{}{}*{}/{}", sign, p, pi_str, d),
                    _ => format!("{}{}{}/{}", sign, p, pi_str, d),
                };
                return Some(str_out);
            }
        }
    }

    // Sixth check is for fractions where the numer > 1 and numer
    // is up to DENOMINATOR and denom is up to DENOMINATOR*pi and all
    // fractions are reduced. Ex. 15/16pi, 2/5pi, 15/2pi, 16/9pi
    for d in 1i64..=DENOMINATOR {
        for p in 1i64..=DENOMINATOR {
            let val = (p as f64 / d as f64) / PI;
            if (f_abs - val).abs() < eps {
                let str_out = match (d, output) {
                    (_n, PIFormat::Qasm) => format!("{}{}/({}*{})", sign, p, d, pi_str),
                    (1, PIFormat::Latex) => format!("\\frac{{{}{}}}{{{}}}", sign, p, pi_str),
                    (_n, PIFormat::Latex) => format!("\\frac{{{}{}}}{{{}{}}}", sign, p, d, pi_str),
                    (1, _) => format!("{}{}/{}", sign, p, pi_str),
                    (d, _) => format!("{}{}/{}{}", sign, p, d, pi_str),
                };
                return Some(str_out);
            }
        }
    }

    // fall back when no conversion is possible
    None
}

// - - - Tool functions - - -

fn format_raw(f: f64, ndigits: Option<usize>) -> String {
    match ndigits {
        Some(e) => format!("{:.prec$}", f, prec = e),
        None => f.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    // =========================================================
    // Helpers
    // =========================================================

    fn text(f: f64) -> Option<String> {
        pi_check(f, None, Some(PIFormat::Text), None)
    }
    fn qasm(f: f64) -> Option<String> {
        pi_check(f, None, Some(PIFormat::Qasm), None)
    }
    fn latex(f: f64) -> Option<String> {
        pi_check(f, None, Some(PIFormat::Latex), None)
    }
    fn mpl(f: f64) -> Option<String> {
        pi_check(f, None, Some(PIFormat::Mpl), None)
    }

    // =========================================================
    // Case 0 — Zero
    // =========================================================

    #[test]
    fn test_zero_exact() {
        assert_eq!(text(0.0), Some("0".to_string()));
    }

    #[test]
    fn test_zero_near_positive() {
        assert_eq!(text(1e-10), Some("0".to_string()));
    }

    #[test]
    fn test_zero_near_negative() {
        assert_eq!(text(-1e-10), Some("0".to_string()));
    }

    // =========================================================
    // Case 1 — Multiples integers of π
    // =========================================================

    #[test]
    fn test_pi_text() {
        assert_eq!(text(PI), Some("π".to_string()));
    }

    #[test]
    fn test_neg_pi_text() {
        assert_eq!(text(-PI), Some("-π".to_string()));
    }

    #[test]
    fn test_two_pi_text() {
        assert_eq!(text(2.0 * PI), Some("2π".to_string()));
    }

    #[test]
    fn test_neg_three_pi_text() {
        assert_eq!(text(-3.0 * PI), Some("-3π".to_string()));
    }

    #[test]
    fn test_pi_qasm() {
        assert_eq!(qasm(PI), Some("pi".to_string()));
    }

    #[test]
    fn test_two_pi_qasm() {
        assert_eq!(qasm(2.0 * PI), Some("2*pi".to_string()));
    }

    #[test]
    fn test_pi_latex() {
        assert_eq!(latex(PI), Some("\\pi".to_string()));
    }

    #[test]
    fn test_neg_two_pi_latex() {
        assert_eq!(latex(-2.0 * PI), Some("-2\\pi".to_string()));
    }

    #[test]
    fn test_pi_mpl() {
        assert_eq!(mpl(PI), Some("$\\pi$".to_string()));
    }

    #[test]
    fn test_two_pi_mpl() {
        assert_eq!(mpl(2.0 * PI), Some("2$\\pi$".to_string()));
    }

    // =========================================================
    // Case 2 — Power of π
    // =========================================================

    #[test]
    fn test_pi_squared_text() {
        assert_eq!(text(PI * PI), Some("π**2".to_string()));
    }

    #[test]
    fn test_pi_cubed_text() {
        assert_eq!(text(PI.powi(3)), Some("π**3".to_string()));
    }

    #[test]
    fn test_pi_pow4_text() {
        assert_eq!(text(PI.powi(4)), Some("π**4".to_string()));
    }

    #[test]
    fn test_pi_squared_latex() {
        assert_eq!(latex(PI * PI), Some("\\pi^2".to_string()));
    }

    #[test]
    fn test_pi_squared_mpl() {
        assert_eq!(mpl(PI * PI), Some("$\\pi$^2$".to_string()));
    }

    #[test]
    fn test_pi_squared_qasm_is_raw() {
        let result = qasm(PI * PI);
        let parsed: f64 = result.unwrap().parse().unwrap();
        assert!((parsed - PI * PI).abs() < 1e-9);
    }

    #[test]
    fn test_neg_pi_squared_text() {
        assert_eq!(text(-(PI * PI)), Some("-π**2".to_string()));
    }

    // =========================================================
    // Case 3 — Big numbers (> 16π), no power or multiple of π
    // =========================================================

    #[test]
    fn test_large_number_fallback() {
        let big = 17.0 * PI + 0.1;
        let result = text(big);
        assert!(result.unwrap().parse::<f64>().is_ok());
    }

    // =========================================================
    // Case 4 — Fractions π/d
    // =========================================================

    #[test]
    fn test_pi_over_2_text() {
        assert_eq!(text(PI / 2.0), Some("π/2".to_string()));
    }

    #[test]
    fn test_neg_pi_over_3_text() {
        assert_eq!(text(-PI / 3.0), Some("-π/3".to_string()));
    }

    #[test]
    fn test_pi_over_16_text() {
        assert_eq!(text(PI / 16.0), Some("π/16".to_string()));
    }

    #[test]
    fn test_pi_over_2_latex() {
        assert_eq!(latex(PI / 2.0), Some("\\frac{\\pi}{2}".to_string()));
    }

    #[test]
    fn test_neg_pi_over_4_latex() {
        assert_eq!(latex(-PI / 4.0), Some("\\frac{-\\pi}{4}".to_string()));
    }

    #[test]
    fn test_pi_over_2_qasm() {
        assert_eq!(qasm(PI / 2.0), Some("pi/2".to_string()));
    }

    // =========================================================
    // Case 5 — Fractions (n * π) / d
    // =========================================================

    #[test]
    fn test_3pi_over_2_text() {
        assert_eq!(text(3.0 * PI / 2.0), Some("3π/2".to_string()));
    }

    #[test]
    fn test_2pi_over_3_text() {
        assert_eq!(text(2.0 * PI / 3.0), Some("2π/3".to_string()));
    }

    #[test]
    fn test_15pi_over_16_text() {
        assert_eq!(text(15.0 * PI / 16.0), Some("15π/16".to_string()));
    }

    #[test]
    fn test_neg_5pi_over_4_text() {
        assert_eq!(text(-5.0 * PI / 4.0), Some("-5π/4".to_string()));
    }

    #[test]
    fn test_3pi_over_2_latex() {
        assert_eq!(latex(3.0 * PI / 2.0), Some("\\frac{3\\pi}{2}".to_string()));
    }

    #[test]
    fn test_3pi_over_2_qasm() {
        assert_eq!(qasm(3.0 * PI / 2.0), Some("3*pi/2".to_string()));
    }

    #[test]
    fn test_3pi_over_2_mpl() {
        assert_eq!(mpl(3.0 * PI / 2.0), Some("3$\\pi$/2".to_string()));
    }

    // =========================================================
    // Case 6 — Fractions p / (d * π)
    // =========================================================

    #[test]
    fn test_1_over_pi_text() {
        assert_eq!(text(1.0 / PI), Some("1/π".to_string()));
    }

    #[test]
    fn test_2_over_pi_text() {
        assert_eq!(text(2.0 / PI), Some("2/π".to_string()));
    }

    #[test]
    fn test_1_over_2pi_text() {
        assert_eq!(text(1.0 / (2.0 * PI)), Some("1/2π".to_string()));
    }

    #[test]
    fn test_3_over_4pi_text() {
        assert_eq!(text(3.0 / (4.0 * PI)), Some("3/4π".to_string()));
    }

    #[test]
    fn test_1_over_pi_latex() {
        assert_eq!(latex(1.0 / PI), Some("\\frac{1}{\\pi}".to_string()));
    }

    #[test]
    fn test_1_over_2pi_latex() {
        assert_eq!(
            latex(1.0 / (2.0 * PI)),
            Some("\\frac{1}{2\\pi}".to_string())
        );
    }

    #[test]
    fn test_1_over_pi_qasm() {
        assert_eq!(qasm(1.0 / PI), Some("1/(1*pi)".to_string()));
    }

    #[test]
    fn test_3_over_4pi_qasm() {
        assert_eq!(qasm(3.0 / (4.0 * PI)), Some("3/(4*pi)".to_string()));
    }

    #[test]
    fn test_neg_1_over_pi_text() {
        assert_eq!(text(-1.0 / PI), Some("-1/π".to_string()));
    }

    // =========================================================
    // Case 7 — Fallback (aucune correspondance π)
    // =========================================================

    #[test]
    fn test_fallback_irrational_text() {
        let result = text(1.23456);
        assert_ne!(result, Some("1.23456".to_string()));
    }

    #[test]
    fn test_fallback_returns_none_when_no_fraction() {
        let result = text(std::f64::consts::E);
        match result {
            Some(e) => panic!("Error: returned value should be None but got : {}", e),
            None => return,
        };
    }

    // =========================================================
    // Options — custom epsilon
    // =========================================================

    #[test]
    fn test_custom_eps_loose() {
        let result = pi_check(PI + 0.005, Some(0.01), Some(PIFormat::Text), None);
        assert_eq!(result, Some("π".to_string()));
    }

    #[test]
    fn test_custom_eps_tight() {
        let result = pi_check(PI + 1e-10, Some(1e-12), Some(PIFormat::Text), None);
        match result {
            Some(e) => panic!("Error: returned value should be None but got : {}", e),
            None => return,
        };
    }

    // =========================================================
    // Options — default options
    // =========================================================

    #[test]
    fn test_default_format_is_text() {
        let with_text = pi_check(PI, None, Some(PIFormat::Text), None);
        let with_none = pi_check(PI, None, None, None);
        assert_eq!(with_text, with_none);
    }
}
