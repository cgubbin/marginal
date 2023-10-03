use num_traits::{Float, FromPrimitive};
use crate::distribution::{Distribution, Covariance};

struct Normal<T> {
    /// Central frequency
    mean: T,
    /// Standard deviation
    standard_deviation: T,
    /// Power raised to, if this is one
    power: usize,
}

impl<T: Float + FromPrimitive + std::fmt::Display> Distribution<T> for Normal<T> {
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
    fn expectation_value(&self) -> T {
        match self.power {
            0 => T::one(),
            1 => self.mean,
            n => {
                self.mean * Self{
                    mean: self.mean, standard_deviation: self.standard_deviation, power: n-1
                }.expectation_value()
                + T::from_usize(n - 1).expect("must fit in `T`") * self.standard_deviation.powi(2) * Self{
                    mean: self.mean, standard_deviation: self.standard_deviation, power: n-2
                }.expectation_value()
            }
        }
    }

    /// The variance of a distribution is
    ///
    /// Var[Z] = E[Z^2] - E[Z]^2
    fn variance(&self) -> T {
        Self{
             mean: self.mean, standard_deviation: self.standard_deviation, power: 2 * self.power
             }.expectation_value()
             - self.expectation_value().powi(2)
    }
}

impl<T: Float + FromPrimitive + std::fmt::Display> Covariance<T> for Normal<T> {
    fn off_diagonal_covariance(&self, other: &Self) -> T {
        if (self.mean == other.mean)
            && (self.standard_deviation == other.standard_deviation) {
            Self {
                mean: self.mean, standard_deviation: self.standard_deviation, power: self.power + other.power
            }.expectation_value() - self.expectation_value() * other.expectation_value()
        } else {
            unimplemented!()
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::distribution::{Covariance, Distribution};
    use super::Normal;
    use rand::prelude::*;
    use proptest::prelude::*;

    #[test]
    fn unit_normal_distribution_has_correct_expectation_value() {
        let mean = 0.0;
        let standard_deviation = 1.0;

        let distribution = Normal { mean, standard_deviation, power: 1 };

        assert_eq!(distribution.expectation_value(), mean);
        assert_eq!(distribution.variance(), 1.0);
    }

    proptest!{
        #[test]
        fn odd_powers_of_unit_normal_distribution_have_correct_expectation_value(
            power in (0..20usize)
                .prop_filter("Values must not divisible by 2",
                     |power| !(0 == power % 2))
        ) {
            let mean = 0.0;
            let standard_deviation = 1.0;
            let distribution = Normal { mean, standard_deviation, power };
            assert_eq!(distribution.expectation_value(), 0.0);
        }
    }

    #[test]
    fn even_powers_of_unit_normal_distribution_have_correct_expectation_value() {
        let mean = 0.0;
        let standard_deviation = 1.0;
        let powers = [2, 4, 6];
        let expected_results = [1.0, 3.0, 15.0];

        for (power, expected) in powers.into_iter().zip(expected_results.into_iter()) {
            let distribution = Normal { mean, standard_deviation, power };
            assert_eq!(distribution.expectation_value(), expected);
        }
    }

    #[test]
    fn normal_distribution_has_correct_expectation_value() {
        let mut rng = rand::thread_rng();
        let mean: f64 = rng.gen();
        let standard_deviation = rng.gen();

        let distribution = Normal { mean, standard_deviation, power: 1 };

        approx::assert_relative_eq!(distribution.expectation_value(), mean);
        approx::assert_relative_eq!(distribution.variance(), standard_deviation.powi(2));
    }

    #[test]
    fn powers_of_normal_distribution_have_correct_expectation_value() {
        let mut rng = rand::thread_rng();
        let mean: f64 = rng.gen();
        let standard_deviation: f64 = rng.gen();

        let powers = [2, 3, 4, 5, 6];
        let expected_results = [
            mean.powi(2) + standard_deviation.powi(2),
            mean.powi(3) + 3. * mean * standard_deviation.powi(2),
            mean.powi(4) + 6. * mean.powi(2) * standard_deviation.powi(2) + 3. * standard_deviation.powi(4),
            mean.powi(5) + 10. * mean.powi(3) * standard_deviation.powi(2) + 15. * mean * standard_deviation.powi(4),
            mean.powi(6) + 15. * mean.powi(4) * standard_deviation.powi(2) + 45. * mean.powi(2) * standard_deviation.powi(4) + 15. * standard_deviation.powi(6),
        ];

        for (power, expected) in powers.into_iter().zip(expected_results.into_iter()) {
            let distribution = Normal { mean, standard_deviation, power };
            approx::assert_relative_eq!(distribution.expectation_value(), expected);
        }
    }

    #[test]
    fn off_diagonal_covariances_of_normal_distribution_with_powers_return_correct() {
        let mut rng = rand::thread_rng();
        let mean: f64 = rng.gen();
        let standard_deviation: f64 = rng.gen();

        let normal = Normal { mean, standard_deviation, power: 1};

        let expected_results = [
            standard_deviation.powi(2),
            2. * mean * standard_deviation.powi(2),
            3. * standard_deviation.powi(2) * (mean.powi(2) + standard_deviation.powi(2)),
            4. * mean * standard_deviation.powi(2) * (mean.powi(2) + 3. * standard_deviation.powi(2)),
            5. * standard_deviation.powi(2) * (mean.powi(4) + 6. * mean.powi(2) * standard_deviation.powi(2) + 3. * standard_deviation.powi(4)),
        ];
        let powers = (1..(expected_results.len() + 1)).collect::<Vec<_>>();

        for (power, expected) in powers.into_iter().zip(expected_results.into_iter()) {
            let other_distribution = Normal { mean, standard_deviation, power };
            approx::assert_relative_eq!(normal.off_diagonal_covariance(&other_distribution), expected);
        }
    }
}
