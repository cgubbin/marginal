use std::ffi::OsString;

pub struct Config<E> {
    pub polynomial_fit_degree: usize,
    pub operating_frequency: E,
}

pub fn args() -> Vec<OsString> {
    todo!()
}
