pub(crate) mod app;
pub(crate) mod args;
mod calibration;
mod cli;
pub(crate) mod config;
pub(crate) mod distribution;
mod mocks;
pub(crate) mod normal;
mod polyfit;

pub use args::Args;

pub(crate) type Result<T> = ::std::result::Result<T, Box<dyn ::std::error::Error>>;
