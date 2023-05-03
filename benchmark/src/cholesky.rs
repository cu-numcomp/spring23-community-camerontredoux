use crate::types;

use super::timeit;
use dyn_stack::{DynStack, GlobalMemBuffer, ReborrowMut};
use faer_core::{Mat, Parallelism};
use std::time::Duration;

pub fn nalgebra_s<T: nalgebra::ComplexField>(sizes: &[usize]) -> Vec<Duration> {
    sizes
        .iter()
        .copied()
        .map(|n| match n {
            1 => {
                let mut c = nalgebra::Matrix1::<T>::zeros();
                for i in 0..n {
                    c[(i, i)] = T::one();
                }

                let time = timeit(|| {
                    nalgebra::linalg::Cholesky::new(c.clone());
                });

                time
            }
            2 => {
                let mut c = nalgebra::Matrix2::<T>::zeros();
                for i in 0..n {
                    c[(i, i)] = T::one();
                }

                let time = timeit(|| {
                    nalgebra::linalg::Cholesky::new(c.clone());
                });

                time
            }
            3 => {
                let mut c = nalgebra::Matrix3::<T>::zeros();
                for i in 0..n {
                    c[(i, i)] = T::one();
                }

                let time = timeit(|| {
                    nalgebra::linalg::Cholesky::new(c.clone());
                });

                time
            }
            4 => {
                let mut c = nalgebra::Matrix4::<T>::zeros();
                for i in 0..n {
                    c[(i, i)] = T::one();
                }

                let time = timeit(|| {
                    nalgebra::linalg::Cholesky::new(c.clone());
                });

                time
            }
            5 => {
                let mut c = nalgebra::Matrix5::<T>::zeros();
                for i in 0..n {
                    c[(i, i)] = T::one();
                }

                let time = timeit(|| {
                    nalgebra::linalg::Cholesky::new(c.clone());
                });

                time
            }
            6 => {
                let mut c = nalgebra::Matrix6::<T>::zeros();
                for i in 0..n {
                    c[(i, i)] = T::one();
                }

                let time = timeit(|| {
                    nalgebra::linalg::Cholesky::new(c.clone());
                });

                time
            }
            7 => {
                let mut c = types::Matrix7::<T>::zeros();
                for i in 0..n {
                    c[(i, i)] = T::one();
                }

                let time = timeit(|| {
                    nalgebra::linalg::Cholesky::new(c.clone());
                });

                time
            }
            8 => {
                let mut c = types::Matrix8::<T>::zeros();
                for i in 0..n {
                    c[(i, i)] = T::one();
                }

                let time = timeit(|| {
                    nalgebra::linalg::Cholesky::new(c.clone());
                });

                time
            }
            9 => {
                let mut c = types::Matrix9::<T>::zeros();
                for i in 0..n {
                    c[(i, i)] = T::one();
                }

                let time = timeit(|| {
                    nalgebra::linalg::Cholesky::new(c.clone());
                });

                time
            }
            10 => {
                let mut c = types::Matrix10::<T>::zeros();
                for i in 0..n {
                    c[(i, i)] = T::one();
                }

                let time = timeit(|| {
                    nalgebra::linalg::Cholesky::new(c.clone());
                });

                time
            }
            11 => {
                let mut c = types::Matrix11::<T>::zeros();
                for i in 0..n {
                    c[(i, i)] = T::one();
                }

                let time = timeit(|| {
                    nalgebra::linalg::Cholesky::new(c.clone());
                });

                time
            }
            12 => {
                let mut c = types::Matrix12::<T>::zeros();
                for i in 0..n {
                    c[(i, i)] = T::one();
                }

                let time = timeit(|| {
                    nalgebra::linalg::Cholesky::new(c.clone());
                });

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
            for i in 0..n {
                c[(i, i)] = T::one();
            }

            let time = timeit(|| {
                nalgebra::linalg::Cholesky::new(c.clone());
            });

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
            for i in 0..n {
                c[(i, i)] = T::one();
            }
            let mut chol = Mat::<T>::zeros(n, n);

            let mut mem = GlobalMemBuffer::new(
                faer_cholesky::llt::compute::cholesky_in_place_req::<T>(
                    n,
                    parallelism,
                    Default::default(),
                )
                .unwrap(),
            );
            let mut stack = DynStack::new(&mut mem);

            let time = timeit(|| {
                chol.as_mut()
                    .cwise()
                    .zip(c.as_ref())
                    .for_each(|dst, src| *dst = src.clone());
                faer_cholesky::llt::compute::cholesky_in_place(
                    chol.as_mut(),
                    parallelism,
                    stack.rb_mut(),
                    Default::default(),
                )
                .unwrap();
            });

            let _ = c;

            time
        })
        .map(Duration::from_secs_f64)
        .collect()
}
