use num_traits::Float;

pub struct Calibration<T: Float> {
    signal: Vec<T>,
    measurement: Vec<T>,
}
