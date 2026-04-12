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
