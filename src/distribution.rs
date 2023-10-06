use itertools::Itertools;
use ndarray::{Array2, arr1};
use num_traits::Float;

pub(crate) trait Distribution<T: Float> {
    fn expectation_value(&self) -> T;
    fn standard_deviation(&self) -> T {
        self.variance().sqrt()
    }
    fn variance(&self) -> T;
}

pub(crate) trait Covariance<T: Float + std::fmt::Debug>: Distribution<T> + Sized + std::fmt::Debug {
    /// Diagonal elements of the covariance matrix
    fn diagonal_covariance(&self, other: &Self) -> [T; 2] {
        let cov_xx = self.variance();
        let cov_yy = other.variance();
        [cov_xx, cov_yy]
    }

    fn off_diagonal_covariance(&self, other: &Self) -> T;

    /// The covariance as Cov[self, self], Cov[self, other], Cov[other, self], Cov[other, other]
    fn covariance(&self, other: &Self) -> [T; 4] {
        let [cov_xx, cov_yy] = self.diagonal_covariance(other);
        let cov_xy = self.off_diagonal_covariance(other);
        // Covariance matrix is Hermitian
        let cov_yx = cov_xy;

        [cov_xx, cov_xy, cov_yx, cov_yy]
    }

    // The matrix Sigma X contains the covariances of all `distributions`
    fn covariance_matrix_sigma_x(distributions: &[Self]) -> Array2<T> {
        let mut sigma_x = Array2::from_diag(
            &arr1(&distributions.iter()
                .map(|dist| dist.variance())
                .collect::<Vec<_>>()
            )
        );

        // Fill the off-diagonal elements
        let indexes = (0..distributions.len()).collect::<Vec<_>>();
        for ((&ii, &jj), (dist, other)) in indexes.iter().tuple_combinations().zip(distributions.iter().tuple_combinations()) {
            let element = dist.off_diagonal_covariance(other);
            sigma_x[[ii, jj]] = element;
            sigma_x[[jj, ii]] = element;
        }

        sigma_x
    }
}


