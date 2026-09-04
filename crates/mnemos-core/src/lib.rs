//! mnemos-core: canonical types, config, errors.
//!
//! Pure logic, minimal deps. Every other crate depends on these
//! definitions — do not change field names without updating all
//! downstream crates.

pub mod config;
pub mod error;
pub mod types;

pub use config::*;
pub use error::*;
pub use types::*;
