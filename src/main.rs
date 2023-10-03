mod distribution;
mod normal;

use std::ops::Range;

use ndarray::{Array2, Array1, ArrayView1, s};
use ndarray_linalg::solve::Inverse;
use ndarray_linalg::types::Lapack;
use num_traits::{Float, FromPrimitive};


/// Calculates the expectation value of the normal distribution raised to `power`
///
/// The expectation value of the `n`th power of a normal distribution is given by
///
/// $$
///     E [X^n] = E[(\mu + \sigma Z)^n] = \sum_{k=0}^n \left(\begin{array}{c}
///     n \\
///     k
///     \end{array}\right)
///     \mu^k \sigma^{n - k} E \left[Z^{n - k}\left]
/// $$
/// Here we used the standard normal distribution with zero mean and unit variance
/// $Z = (X - \mu)/\sigma. The mean of the distribution is notated $\mu$ and the
/// standard deviation as $\sigma$.
fn expectation_value<T: Float + FromPrimitive>(n: usize, mu: T, sigma: T) -> T {
    match n {
        0 => T::one(),
        1 => mu,
        _ => mu * expectation_value(n-1, mu, sigma) + T::from_usize(n - 1).expect("must fit in `T`") * sigma.powi(2) * expectation_value(n-2, mu, sigma),
    }
}

struct SolutionCoefficients<T: Sized, const ORDER: usize> {
    a: [T; ORDER],
}

impl<T: Sized, const ORDER: usize> SolutionCoefficients<T, ORDER> {
    fn order(&self) -> usize {
        self.a.len()
    }
}

#[derive(Debug)]
struct Problem<T: Float + FromPrimitive + std::fmt::Debug> {
    sigma_x: Array2<T>,
    mu_x_tilde: Array1<T>,
    beta_0: T,
    beta: Array1<T>,
    sigma_y_x: Array1<T>,
    mu_y: T,
}

impl<T: Lapack + Float + FromPrimitive + std::fmt::Debug> Problem<T> {
    fn build<const ORDER: usize>(coefficients: SolutionCoefficients<T, ORDER>, mu: T, sigma: T) -> Self {
        let order = coefficients.order();
        let number_of_cumulants = (order - 1) * (order - 1);
        let mu_x_tilde = (1..=(number_of_cumulants))
            .map(|n| expectation_value(n, mu, sigma))
            .collect::<Array1<T>>();
        let mut sigma_x = Array2::zeros((order - 1, order - 1));
        for ii in 0..(order-1) {
            for jj in 0..(order-1) {
                sigma_x[[ii, jj]] = mu_x_tilde[[ii+jj+1]] - mu_x_tilde[[ii]] * mu_x_tilde[[jj]];
            }
        }

        let mut beta = Array1::zeros(order - 1);
        for ii in 0..(order - 1) {
            beta[[ii]] = coefficients.a[ii + 1];
        }
        let beta_0 = coefficients.a[0];

        let mut sigma_y_x = Array1::zeros(order - 1);
        for ii in 0..(order - 1) {
            let row = sigma_x.slice(s![ii, ..]);
            let val: T = row.iter().zip(beta.iter()).fold(T::zero(), |a, (&x, &y)| a + (x * y));
            sigma_y_x[[ii]] = val;
        }

        let mu_y = beta_0 + beta.iter().zip(mu_x_tilde.iter()).map(|(&beta, &mu)| beta * mu).fold(T::zero(), |a, b| a + b);

        Self { mu_x_tilde, sigma_x, beta, beta_0, sigma_y_x, mu_y }
    }

    fn mu_x(&self) -> ArrayView1<T> {
        self.mu_x_tilde.slice(s![..self.sigma_x.dim().0])
    }

    fn bounds(&self, z_score: T) -> Range<T> {
        let sigma = Float::sqrt(self.sigma_y_x
            .dot(&self.sigma_x.inv().expect("inverse failed").dot(&self.sigma_y_x))
        );
        (self.mu_y-z_score * sigma)..(self.mu_y + z_score + sigma)
    }
}

fn main() {
    let a = [
        -1.01461546e-1,
        -1.81372854e3,
        5.25687728e3,
        -9.32865392e4,
        -3.45774589e5,
        -4.24001854e5
    ];

    let mu = -0.12128476455755798;
    let sigma = 0.00104241;

    let coefficients = SolutionCoefficients{ a };
    let problem = Problem::build(coefficients, mu, sigma);
    println!("{:?}", problem.bounds(1.96));
}


#[cfg(test)]
mod test {

    #[test]
    fn test_expectation() {
        let mu: f64 = 3.0;
        let sigma: f64 = 0.2;

        let results = vec![
            1.0,
            mu,
            mu.powi(2) + sigma.powi(2),
            mu.powi(3) + 3.0 * mu * sigma.powi(2),
            mu.powi(4) + 6.0 * mu.powi(2) * sigma.powi(2) + 3. * sigma.powi(4),
            mu.powi(5) + 10. * mu.powi(3) * sigma.powi(2) + 15. * mu * sigma.powi(4),
            mu.powi(6) + 15. * mu.powi(4) * sigma.powi(2) + 45. * mu.powi(2) * sigma.powi(4) + 15. * sigma.powi(6),
        ];

        for (n, result) in results.into_iter().enumerate() {
            let value = super::expectation_value(n, mu, sigma);
            println!("{n}, {result:.2}, {value:.2}");
        }
    }

}
