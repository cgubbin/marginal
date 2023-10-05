pub(crate) mod app;
pub mod args;
mod calibration;
mod cli;
pub(crate) mod config;
pub(crate) mod distribution;
mod mocks;
pub(crate) mod normal;
mod polyfit;

pub type Result<T> = ::std::result::Result<T, Box<dyn ::std::error::Error>>;

