//! Basic ingestion example.
//!
//! Run with:
//!   cargo run --example basic_ingest --features mnemos-app/cli

use mnemos_core::MnemosConfig;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = MnemosConfig::default();
    println!("MNEMOS config: {:?}", config);
    Ok(())
}
