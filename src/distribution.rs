use num_traits::Float;

pub(crate) trait Distribution<T: Float> {
    fn expectation_value(&self) -> T;
    fn standard_deviation(&self) -> T {
        self.variance().sqrt()
    }
    fn variance(&self) -> T;
}

pub(crate) trait Covariance<T: Float>: Distribution<T> {
    /// The covariance as Cov[self, self], Cov[self, other], Cov[other, self], Cov[other, other]
    fn diagonal_covariance(&self, other: &Self) -> [T; 2] {
        let cov_xx = self.variance();
        let cov_yy = other.variance();
        [cov_xx, cov_yy]
    }

    fn off_diagonal_covariance(&self, other: &Self) -> T;

    fn covariance(&self, other: &Self) -> [T; 4] {
        let [cov_xx, cov_yy] = self.diagonal_covariance(other);
        let cov_xy = self.off_diagonal_covariance(other);
        // Covariance matrix is Hermitian
        let cov_yx = cov_xy;

        [cov_xx, cov_xy, cov_yx, cov_yy]
    }
}
