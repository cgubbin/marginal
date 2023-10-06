use std::collections::HashMap;

use ndarray_linalg::Scalar;

use crate::Result;
use crate::calibration::Gas;
use crate::calibration::Measurement;
use crate::calibration::Sensor;

fn estimate<E: Scalar>(
    sensors: Vec<Sensor<E>>,
    measurement: HashMap<Gas, Measurement<E>>,
) -> Result<()> {
    // TODO: Runtime checks, we will enforce using the type system at a later point
    assert!(
        sensors.len() > 0,
        "at least one sensors must be provided to reconstruct"
    );
    assert!(
        sensors.len() == measurement.len(),
        "all sensors must have an associated measurement at each timestep"
    );
    for sensor in sensors.iter() {
        assert!(
            measurement.get(sensor.target()).is_some(),
            "the sensors and measurements lists are inconsistent"
        );
    }

    match sensors.len() {
        1 => {
            let target = sensors[0].target();
            let _estimate = estimate_single_gas(
                &sensors[0], measurement.get(&target).unwrap()
            );
            Ok(())
        }
        _ => unimplemented!("no multigas impl"),
    }
}


fn estimate_single_gas<E: Scalar>(
    sensor: &Sensor<E>,
    measurement: &Measurement<E>,
) -> Result<E> {
    todo!()

}
