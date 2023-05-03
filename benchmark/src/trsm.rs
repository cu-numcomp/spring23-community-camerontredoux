use crate::types;

use super::timeit;
use faer_core::{Conj, Mat, Parallelism};
use std::time::Duration;

pub fn nalgebra_s<T: nalgebra::ComplexField>(sizes: &[usize]) -> Vec<Duration> {
    sizes
        .iter()
        .copied()
        .map(|n| match n {
            1 => {
                let mut c = nalgebra::Matrix1::<T>::zeros();
                let a = nalgebra::Matrix1::<T>::zeros();

                let time = timeit(|| {
                    a.solve_lower_triangular_with_diag_mut(&mut c, T::one());
                });

                let _ = c;

                time
            }
            2 => {
                let mut c = nalgebra::Matrix2::<T>::zeros();
                let a = nalgebra::Matrix2::<T>::zeros();

                let time = timeit(|| {
                    a.solve_lower_triangular_with_diag_mut(&mut c, T::one());
                });

                let _ = c;

                time
            }
            3 => {
                let mut c = nalgebra::Matrix3::<T>::zeros();
                let a = nalgebra::Matrix3::<T>::zeros();

                let time = timeit(|| {
                    a.solve_lower_triangular_with_diag_mut(&mut c, T::one());
                });

                let _ = c;

                time
            }
            4 => {
                let mut c = nalgebra::Matrix4::<T>::zeros();
                let a = nalgebra::Matrix4::<T>::zeros();

                let time = timeit(|| {
                    a.solve_lower_triangular_with_diag_mut(&mut c, T::one());
                });

                let _ = c;

                time
            }
            5 => {
                let mut c = nalgebra::Matrix5::<T>::zeros();
                let a = nalgebra::Matrix5::<T>::zeros();

                let time = timeit(|| {
                    a.solve_lower_triangular_with_diag_mut(&mut c, T::one());
                });

                let _ = c;

                time
            }
            6 => {
                let mut c = nalgebra::Matrix6::<T>::zeros();
                let a = nalgebra::Matrix6::<T>::zeros();

                let time = timeit(|| {
                    a.solve_lower_triangular_with_diag_mut(&mut c, T::one());
                });

                let _ = c;

                time
            }
            7 => {
                let mut c = types::Matrix7::<T>::zeros();
                let a = types::Matrix7::<T>::zeros();

                let time = timeit(|| {
                    a.solve_lower_triangular_with_diag_mut(&mut c, T::one());
                });

                let _ = c;

                time
            }
            8 => {
                let mut c = types::Matrix8::<T>::zeros();
                let a = types::Matrix8::<T>::zeros();

                let time = timeit(|| {
                    a.solve_lower_triangular_with_diag_mut(&mut c, T::one());
                });

                let _ = c;

                time
            }
            9 => {
                let mut c = types::Matrix9::<T>::zeros();
                let a = types::Matrix9::<T>::zeros();

                let time = timeit(|| {
                    a.solve_lower_triangular_with_diag_mut(&mut c, T::one());
                });

                let _ = c;

                time
            }
            10 => {
                let mut c = types::Matrix10::<T>::zeros();
                let a = types::Matrix10::<T>::zeros();

                let time = timeit(|| {
                    a.solve_lower_triangular_with_diag_mut(&mut c, T::one());
                });

                let _ = c;

                time
            }
            11 => {
                let mut c = types::Matrix11::<T>::zeros();
                let a = types::Matrix11::<T>::zeros();

                let time = timeit(|| {
                    a.solve_lower_triangular_with_diag_mut(&mut c, T::one());
                });

                let _ = c;

                time
            }
            12 => {
                let mut c = types::Matrix12::<T>::zeros();
                let a = types::Matrix12::<T>::zeros();

                let time = timeit(|| {
                    a.solve_lower_triangular_with_diag_mut(&mut c, T::one());
                });

                let _ = c;

                time
            }
            _ => 0.0,
        })
        .map(Duration::from_secs_f64)
        .collect()
}

pub fn nalgebra<T: nalgebra::ComplexField>(sizes: &[usize]) -> Vec<Duration> {
    sizes
        .iter()
        .copied()
        .map(|n| {
            let mut c = nalgebra::DMatrix::<T>::zeros(n, n);
            let a = nalgebra::DMatrix::<T>::zeros(n, n);

            let time = timeit(|| {
                a.solve_lower_triangular_with_diag_mut(&mut c, T::one());
            });

            let _ = c;

            time
        })
        .map(Duration::from_secs_f64)
        .collect()
}

pub fn faer<T: faer_core::ComplexField>(
    sizes: &[usize],
    parallelism: Parallelism,
) -> Vec<Duration> {
    sizes
        .iter()
        .copied()
        .map(|n| {
            let mut c = Mat::<T>::zeros(n, n);
            let a = Mat::<T>::zeros(n, n);

            let time = timeit(|| {
                faer_core::solve::solve_unit_lower_triangular_in_place(
                    a.as_ref(),
                    Conj::No,
                    c.as_mut(),
                    Conj::No,
                    parallelism,
                );
            });

            let _ = c;

            time
        })
        .map(Duration::from_secs_f64)
        .collect()
}
