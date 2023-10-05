use std::ffi::OsStr;
use std::fs;
use std::path::PathBuf;
use std::{collections::HashMap, marker::PhantomData};

use ndarray::ScalarOperand;
use ndarray_linalg::{Scalar, Lapack};
use num_traits::real::Real;
use serde::{Deserialize, de::DeserializeOwned, Serialize};

use crate::Result;
use crate::polyfit::{PolyfitResult, polyfit};

struct Config<E> {
    polynomial_fit_degree: usize,
    operating_frequency: E,
}

fn build<E: Lapack + Scalar + ScalarOperand + Real>(working_directory: PathBuf, config: Config<E>) -> Result<Vec<Sensor<E>>> {

    let mut sensors = vec![];
    for subdir in fs::read_dir(working_directory.join("calibration"))? {
        println!("working in {subdir:?}");
        let subdir = subdir?;
        let path = subdir.path();
        let sensor_calibration_data: SensorBuilder<E, Set> = process(path, &config)?;
        sensors.push(sensor_calibration_data);
    }

    // Build the sensors
    let sensors = sensors
        .into_iter()
        .map(|builder| builder.build())
        .collect::<Result<Vec<Sensor<E>>>>()?;


    Ok(sensors)
}

#[derive(Deserialize, Serialize)]
struct SensorData<E> {
    noise_equivalent_power: E,
}

fn process<E: Scalar>(path: PathBuf, config: &Config<E>) -> Result<SensorBuilder<E, Set>> {
    let target = Gas(path.file_stem().unwrap().to_string_lossy().into_owned()); // The target gas is the directory name
    println!("Working on target {target:?}");
    let sensor_data_file = path.join("sensor.toml");
    let sensor_data = fs::read_to_string(&sensor_data_file)?;
    let sensor_data: SensorData<E> = toml::from_str(&sensor_data)?;
    println!("Successfully read sensor file");

    // Paths to crosstalk
    let csv_file_paths = fs::read_dir(&path)?
        .into_iter()
        .filter(|r| r.is_ok())
        .map(|r| r.unwrap())
        // Map the directory entries to paths
        .map(|dir_entry| dir_entry.path())
        // Filter out all paths with extensions other than `csv`
        .filter_map(|path| {
            if path.extension().map_or(false, |ext| ext == "csv") {
                Some(path)
            } else {
                None
            }
        })
        .filter(|path| {
            path.file_stem().and_then(OsStr::to_str).map_or(false, |stem| stem != &target.0)
        });



    let mut builder: SensorBuilder<E, Unset> = SensorBuilder::new(
        target.clone(),
        sensor_data.noise_equivalent_power,
        config.operating_frequency,
        config.polynomial_fit_degree,
    );

    for csv_file_path in csv_file_paths {
        println!("reading {csv_file_path:?}");
        let crosstalk_data = CalibrationData::from_file(&csv_file_path)?;
        builder = builder.with_crosstalk(crosstalk_data);
    }

    let path_to_calibration_csv = path.join(format!("{}.csv", target.0));
    let calibration_data = CalibrationData::from_file(&path_to_calibration_csv)?;
    let builder = builder.with_calibration(calibration_data);

    Ok(builder)
}

#[cfg(test)]
mod tests {
    use ndarray_rand::rand::{Rng, SeedableRng};
    use rand_isaac::Isaac64Rng;
    use tempdir::TempDir;

    use crate::calibration::SensorData;

    use super::Config;


    #[test]
    fn test_traversal() {
        // Arrange
        let tmp_dir = TempDir::new("test_traversal").unwrap();
        let working_dir = tmp_dir.path();
        let calibration_dir = working_dir.join("calibration");
        std::fs::create_dir(&calibration_dir).unwrap();

        let gases = ["H2O", "CO2"];
        let seed = 40;
        let mut rng = Isaac64Rng::seed_from_u64(seed);

        for gas in gases {
            let gas_dir = calibration_dir.join(gas);
            std::fs::create_dir(&gas_dir).unwrap();
            let sensor_data = SensorData { noise_equivalent_power: rng.gen::<f64>() };
            std::fs::write(gas_dir.join("sensor.toml"), toml::to_string(&sensor_data).unwrap())
                .unwrap();
        }

        let polynomial_degree = 4;

        let coeffs = (0..=polynomial_degree)
            .map(|_| rng.gen::<f64>())
            .collect::<Vec<_>>();

        // We consider emergent intensities to be 1.0, reference to be 1.0, and only change the
        // recorded intensity. our signal i
        let num_points = 255;
        let concentrations = (0..num_points)
            .map(|n| n as f64 / num_points as f64)
            .collect::<Vec<_>>();
        let values = concentrations.iter()
            .map(|x| coeffs.iter().enumerate().map(|(ii, c)| c * x.powi(ii as i32)).fold(0., |a, b| a + b))
            .map(|x| 10f64.powf(x))
            .collect::<Vec<_>>();


        #[derive(serde::Serialize)]
        struct Row(f64, f64, f64, f64, f64);

        for target in gases {
            for gas in gases {
                let mut wtr = csv::Writer::from_path(calibration_dir.join(target).join(format!("{gas}.csv"))).unwrap();
                for (c, v) in concentrations.iter().zip(values.iter()) {
                    let row = Row(*c, *v, 1.0, 1.0, 1.0);
                    wtr.serialize(&row).unwrap();
                }
            }
        }


        let config = Config {
            polynomial_fit_degree: 4,
            operating_frequency: 10.0,
        };
        let res = super::build::<f64>(working_dir.to_path_buf(), config).unwrap();
    }
}


#[derive(Clone, Hash, PartialEq, Eq, Debug)]
struct Gas(String);

#[derive(Debug)]
struct Sensor<E: Scalar + std::fmt::Debug> {
    /// The molecule the sensor is designed to detect
    target: Gas,
    /// Frequency of operation
    operation_frequency: E,
    /// The noise equivalent power of the photodetector in nano-Watt
    noise_equivalent_power: E,
    /// Calibration Curve
    calibration: PolyfitResult<E>,
    /// Crosstalk curves, may or may not be provided
    crosstalk: HashMap<Gas, PolyfitResult<E>>,
}

enum Set {}
enum Unset {}

struct SensorBuilder<E: Scalar, N> {
    target: Gas,
    operation_frequency: E,
    noise_equivalent_power: E,
    polynomial_degree: usize,
    raw_calibration_data: Vec<CalibrationData<E>>,
    phantom_data: PhantomData<N>,
}

impl<E: Scalar, N> SensorBuilder<E, N> {
    fn new(target: Gas, noise_equivalent_power: E, operation_frequency: E, polynomial_degree: usize) -> Self {
        Self {
            target,
            noise_equivalent_power,
            operation_frequency,
            polynomial_degree,
            raw_calibration_data: vec![],
            phantom_data: PhantomData
        }
    }

    fn with_crosstalk(mut self, calibration_data: CalibrationData<E>) -> Self {
        self.raw_calibration_data.push(calibration_data);
        self
    }
}

impl<E: Scalar> SensorBuilder<E, Unset> {
    fn with_calibration(mut self, calibration_data: CalibrationData<E>) -> SensorBuilder<E, Set> {
        self.raw_calibration_data.push(calibration_data);
        SensorBuilder {
            target: self.target,
            noise_equivalent_power: self.noise_equivalent_power,
            operation_frequency: self.operation_frequency,
            polynomial_degree: self.polynomial_degree,
            raw_calibration_data: self.raw_calibration_data,
            phantom_data: PhantomData
        }
    }
}

impl<E: Lapack + Real + Scalar + ScalarOperand> SensorBuilder<E, Set> {
    fn build(self) -> Result<Sensor<E>> {
        let mut crosstalk = HashMap::new();
        let mut calibration = None;
        for calibration_data in self.raw_calibration_data {
            let fit = generate_fit(&calibration_data, self.noise_equivalent_power, self.operation_frequency, self.polynomial_degree)?;

            if calibration_data.gas == self.target {
                calibration = Some(fit);
            } else {
                crosstalk.insert(calibration_data.gas.clone(), fit);
            }
        }

        Ok(Sensor {
            target: self.target,
            noise_equivalent_power: self.noise_equivalent_power,
            operation_frequency: self.operation_frequency,
            calibration: calibration.unwrap(),
            crosstalk,
        })
    }
}

fn generate_fit<E: Lapack + Real + Scalar + ScalarOperand>(
    calibration_data: &CalibrationData<E>,
    noise_equivalent_power: E,
    operating_frequency: E,
    degree: usize
) -> Result<PolyfitResult<E>> {
    let data = calibration_data.generate_fitting_data(noise_equivalent_power, operating_frequency);
    let fit = polyfit(
        &data.x,
        &data.y,
        degree,
        Some(&data.w),
        crate::polyfit::Scaling::Unscaled
    )?;
    Ok(fit)
}


struct CalibrationData<E: Scalar> {
    gas: Gas,
    concentration: Vec<E>,
    raw_measurements: Vec<Measurement<E>>,
}


impl<E: Scalar + Copy> CalibrationData<E>
where
    E: DeserializeOwned,
{
    /// Create a `CalibrationData` from an on-disk representation
    fn from_file(filepath: &PathBuf) -> Result<Self> {
        if !filepath.exists() {
            return Err("requested file not found".into());
        }

        let gas = filepath.file_stem()
            .expect("filestem missing")
            .to_str()
            .expect("failed to convert stem to string")
            .to_owned();

        let file = fs::read(filepath)?;
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_reader(&file[..]);

        let mut concentration = vec![];
        let mut raw_measurements = vec![];

        #[derive(Deserialize)]
        struct Row<E>(E, E, E, E, E);

        for result in rdr.deserialize() {
            let record: Row<E> = result?;
            concentration.push(record.0);
            let measurement = Measurement {
                raw_signal: record.1,
                raw_reference: record.2,
                emergent_signal: record.3,
                emergent_reference: record.4,
            };
            raw_measurements.push(measurement);
        }

        Ok(Self {
            gas: Gas(gas), concentration, raw_measurements
        })
    }
}

struct Measurement<E: Scalar> {
    raw_signal: E,
    raw_reference: E,
    emergent_signal: E,
    emergent_reference: E,
}

impl<E: Real + Scalar> Measurement<E> {
    fn scaled(&self) -> E {
        (self.raw_signal / self.emergent_signal
            * self.emergent_reference / self.emergent_signal).log10()
    }

    // The weights are the inverse of the variance of the measurement
    fn weight(&self, noise_equivalent_power: E, operation_frequency: E) -> E {
        let standard_deviation = noise_equivalent_power * Scalar::sqrt(operation_frequency)
            * (self.raw_reference + self.raw_reference)
            / Scalar::powi(self.raw_signal, 2);

        E::one() / Scalar::powi(standard_deviation, 2)

    }
}


struct PolyFitInput<E> {
    x: Vec<E>,
    y: Vec<E>,
    w: Vec<E>,
}

impl<E: Real + Scalar> CalibrationData<E> {
    fn generate_fitting_data(&self, noise_equivalent_power: E, operation_frequency: E) -> PolyFitInput<E> {
        PolyFitInput {
            x: self.concentration.clone(),
            y: self.raw_measurements.iter().map(|measurement| measurement.scaled()).collect(),
            w: self.raw_measurements.iter().map(|measurement| measurement.weight(noise_equivalent_power, operation_frequency)).collect(),
        }
    }
}
