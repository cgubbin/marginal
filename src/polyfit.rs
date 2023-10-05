use std::ops::MulAssign;

use ndarray::{Array, Array2, Array1, s, ScalarOperand, Axis, Ix0};
use ndarray_linalg::{LeastSquaresSvd, Inverse, Lapack, Scalar, LeastSquaresResult};

use crate::Result;

#[derive(PartialEq, Eq)]
pub enum Scaling {
    Scaled,
    Unscaled,
}

#[derive(Debug)]
pub struct PolyfitResult<E: Scalar> {
    solution: Array1<E>,
    covariance: Array2<E>,
    singular_values: Array1<E::Real>,
    rank: i32,
    residual_sum_of_squares: Option<Array<E::Real, Ix0>>,
}

pub fn polyfit<T: Scalar + Lapack + ScalarOperand + MulAssign + Copy>(
    x: &[T],
    y: &[T],
    degree: usize,
    maybe_weights: Option<&[T]>,
    covariance_scaling: Scaling,
) -> Result<PolyfitResult<T>> {
    let vander = vandermonde(x, degree)?;
    let mut lhs: Array2<T> = vander.to_owned();
    let mut rhs: Array1<T> = Array::from_iter(y.into_iter().copied()).into_shape(x.len())?;
    if let Some(weights) = maybe_weights {
        let weights: Array1<T> = Array::from_iter(weights.into_iter().copied()).into_shape(x.len())?;
        rhs = rhs * &weights;

        for (ii, weight) in weights.iter().enumerate() {
            let mut slice = lhs.slice_mut(s![ii, ..]);
            slice *= *weight;
        }
    }

    let scaling: Array1<T> = lhs
        .mapv(|val| val.powi(2))
        .sum_axis(Axis(0))
        .mapv(|val| val.sqrt());

    lhs = lhs / &scaling;
    let result = lhs.least_squares(&rhs)?;
    let solution = (&result.solution.t() / &scaling).t().to_owned();


    let covariance = (lhs.t().dot(&lhs)).inv()?;
    let outer_prod_of_scaling = outer_product(&scaling, &scaling)?;
    let mut covariance = covariance / outer_prod_of_scaling;
    if covariance_scaling == Scaling::Scaled {
        let factor = result.residual_sum_of_squares.as_ref().unwrap().mapv(|re| T::from_real(re) / T::from(x.len() - degree).unwrap());
        covariance = covariance * factor;
    };


    Ok(PolyfitResult {
        solution,
        covariance,
        singular_values: result.singular_values,
        rank: result.rank,
        residual_sum_of_squares: result.residual_sum_of_squares,
    })
}

fn outer_product<T: Scalar>(
    a: &Array1<T>,
    b: &Array1<T>,
) -> Result<Array2<T>> {
    let a: Array2<T> = a.clone().into_shape((a.len(), 1))?;
    let b: Array2<T> = b.clone().into_shape((1, b.len()))?;

    Ok(ndarray::linalg::kron(&a, &b))
}

fn vandermonde<T: Scalar + Copy>(
    x: &[T],
    degree: usize,
) -> Result<Array2<T>> {
    let vals = x.iter()
        .map(|xi| (0..=degree).map(|i| xi.powi(i as i32)))
        .flatten();

    Ok(Array::from_iter(vals).into_shape((x.len(), degree + 1))?)
}

#[cfg(test)]
mod test {
    use crate::polyfit::Scaling;
    use super::{polyfit, vandermonde, outer_product};

    use itertools::Itertools;
    use ndarray_linalg::Determinant;

    use ndarray::Array;
    use ndarray_rand::{rand::Rng, RandomExt};
    use ndarray_rand::rand::SeedableRng;
    use ndarray_rand::rand_distr::Uniform;
    use rand_isaac::isaac64::Isaac64Rng;

    #[test]
    fn vandermonde_matrices_are_generated_correctly() {
        let seed = 40;
        let mut rng = Isaac64Rng::seed_from_u64(seed);
        let num_data_points = 10;
        let degree = 5;

        let data_points = (0..num_data_points).map(|_| rng.gen()).collect::<Vec<f64>>();

        let vandermonde = vandermonde(&data_points, degree).unwrap();

        for (ii, data_point) in data_points.iter().enumerate() {
            for jj in 0..=degree {
                let expected = data_point.powi(jj as i32);
                let actual = vandermonde[[ii, jj]];
                approx::assert_relative_eq!(expected, actual);
            }
        }
    }

    #[test]
    fn determinant_of_square_vandermonde_matrix_equals_product_of_differences() {
        let dim = 5;
        let seed = 40;
        let mut rng = Isaac64Rng::seed_from_u64(seed);
        let data_points = (0..dim).map(|_| rng.gen()).collect::<Vec<f64>>();

        let vandermonde = vandermonde(&data_points, dim - 1).unwrap();
        let determinant = vandermonde.det().unwrap();

        let product_of_differences: f64 = data_points.iter()
            .combinations(2)
            .map(|vals| vals[0] - vals[1])
            .product();

        approx::assert_relative_eq!(determinant, product_of_differences);
    }

    #[test]
    fn outer_products_are_generated_correctly() {
        let seed = 40;
        let mut rng = Isaac64Rng::seed_from_u64(seed);
        let len_a = rng.gen::<u8>() as usize;
        let len_b = rng.gen::<u8>() as usize;
        let a = Array::random_using(len_a, Uniform::new(0., 10.), &mut rng);
        let b = Array::random_using(len_b, Uniform::new(0., 10.), &mut rng);

        let outer = outer_product(&a, &b).unwrap();

        for ii in 0..len_a {
            for jj in 0..len_b {
                approx::assert_relative_eq!(outer[[ii, jj]], a[ii] * b[jj]);
            }
        }

    }

    #[test]
    fn quadratic_polynomials_are_fit_correctly() {
        let degree = 2;
        let seed = 40;
        let mut rng = Isaac64Rng::seed_from_u64(seed);
        let num_samples = rng.gen_range(10..255);
        let coeffs = (0..=degree).map(|_| rng.gen()).collect::<Vec<f64>>();
        let x = (0..num_samples).map(|n| n as f64).collect::<Vec<_>>();
        let y = x.iter().map(|x| coeffs.iter().enumerate().map(|(ii, ci)| ci * x.powi(ii as i32)).sum()).collect::<Vec<_>>();

        let result = polyfit(&x, &y, degree, None, Scaling::Scaled).unwrap();

        for (coeff, fitted) in coeffs.into_iter().zip(result.solution.into_iter()) {
            approx::assert_relative_eq!(coeff, fitted, max_relative=1e-10);
        }
    }

    #[test]
    fn cubic_polynomials_are_fit_correctly() {
        let degree = 3;
        let seed = 40;
        let mut rng = Isaac64Rng::seed_from_u64(seed);
        let num_samples = rng.gen_range(10..255);
        let coeffs = (0..=degree).map(|_| rng.gen()).collect::<Vec<f64>>();
        let x = (0..num_samples).map(|n| n as f64).collect::<Vec<_>>();
        let y = x.iter().map(|x| coeffs.iter().enumerate().map(|(ii, ci)| ci * x.powi(ii as i32)).sum()).collect::<Vec<_>>();

        let result = polyfit(&x, &y, degree, None, Scaling::Scaled).unwrap();

        for (coeff, fitted) in coeffs.into_iter().zip(result.solution.into_iter()) {
            approx::assert_relative_eq!(coeff, fitted, max_relative=1e-10);
        }
    }
}

