use qiskit_circuit::circuit_data::CircuitDataError;
use rand::distr::weighted::Error as DistroError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum EvolutionError {
    /// An invalid order value passed to the main method
    #[error["Suzuki product formulae are symmetric and therefore only defined \
            for when the order is 1 or even, not {0}."]]
    OrderSymmetry(u32),

    /// A general error for terms reordering error
    #[error["Error ocurred when trying to reorder terms: {0}"]]
    TermsReorder(String),

    /// A general error when trying to build circuit from generated instructions
    #[error["Failed building circuit"]]
    CircuitBuild(#[from] CircuitDataError),

    /// Complex value obtained from real approximation
    #[error["Encountered complex value {0}, but expected real."]]
    UnexpectedImaginaryPart(f64),

    /// Couldn't generate weighted distribution
    #[error["Failed creating weighted distribution"]]
    DistributionError(#[from] DistroError),
}
