use super::timeit;
use faer_core::{Conj, Mat, Parallelism};
// use simba::simd::{f32x2, f32x4};
use std::time::Duration;

use super::types;

pub fn nalgebra_s<T: nalgebra::ComplexField>(sizes: &[usize]) -> Vec<Duration> {
    sizes
        .iter()
        .copied()
        .map(|n| match n {
            1 => {
                let mut c = nalgebra::Matrix1::<T>::zeros();
                let a = nalgebra::Matrix1::<T>::zeros();
                let b = nalgebra::Matrix1::<T>::zeros();

                let time = timeit(|| {
                    a.mul_to(&b, &mut c);
                });

                time
            }
            2 => {
                let mut c = nalgebra::Matrix2::<T>::zeros();
                let a = nalgebra::Matrix2::<T>::zeros();
                let b = nalgebra::Matrix2::<T>::zeros();

                let time = timeit(|| {
                    a.mul_to(&b, &mut c);
                });

                time
            }
            3 => {
                let mut c = nalgebra::Matrix3::<T>::zeros();
                let a = nalgebra::Matrix3::<T>::zeros();
                let b = nalgebra::Matrix3::<T>::zeros();

                let time = timeit(|| {
                    a.mul_to(&b, &mut c);
                });

                time
            }
            4 => {
                let mut c = nalgebra::Matrix4::<T>::zeros();
                let a = nalgebra::Matrix4::<T>::zeros();
                let b = nalgebra::Matrix4::<T>::zeros();

                let time = timeit(|| {
                    a.mul_to(&b, &mut c);
                });

                time
            }
            5 => {
                let mut c = nalgebra::Matrix5::<T>::zeros();
                let a = nalgebra::Matrix5::<T>::zeros();
                let b = nalgebra::Matrix5::<T>::zeros();

                let time = timeit(|| {
                    a.mul_to(&b, &mut c);
                });

                time
            }
            6 => {
                let mut c = nalgebra::Matrix6::<T>::zeros();
                let a = nalgebra::Matrix6::<T>::zeros();
                let b = nalgebra::Matrix6::<T>::zeros();

                let time = timeit(|| {
                    a.mul_to(&b, &mut c);
                });

                time
            }
            7 => {
                let mut c = types::Matrix7::<T>::zeros();
                let a = types::Matrix7::<T>::zeros();
                let b = types::Matrix7::<T>::zeros();

                let time = timeit(|| {
                    a.mul_to(&b, &mut c);
                });

                time
            }
            8 => {
                let mut c = types::Matrix8::<T>::zeros();
                let a = types::Matrix8::<T>::zeros();
                let b = types::Matrix8::<T>::zeros();

                let time = timeit(|| {
                    a.mul_to(&b, &mut c);
                });

                time
            }
            9 => {
                let mut c = types::Matrix9::<T>::zeros();
                let a = types::Matrix9::<T>::zeros();
                let b = types::Matrix9::<T>::zeros();

                let time = timeit(|| {
                    a.mul_to(&b, &mut c);
                });

                time
            }
            10 => {
                let mut c = types::Matrix10::<T>::zeros();
                let a = types::Matrix10::<T>::zeros();
                let b = types::Matrix10::<T>::zeros();

                let time = timeit(|| {
                    a.mul_to(&b, &mut c);
                });

                time
            }
            11 => {
                let mut c = types::Matrix11::<T>::zeros();
                let a = types::Matrix11::<T>::zeros();
                let b = types::Matrix11::<T>::zeros();

                let time = timeit(|| {
                    a.mul_to(&b, &mut c);
                });

                time
            }
            12 => {
                let mut c = types::Matrix12::<T>::zeros();
                let a = types::Matrix12::<T>::zeros();
                let b = types::Matrix12::<T>::zeros();

                let time = timeit(|| {
                    a.mul_to(&b, &mut c);
                });

                time
            }
            _ => 0.0,
        })
        .map(Duration::from_secs_f64)
        .collect()
}

pub fn nalgebra_d<T: nalgebra::ComplexField>(sizes: &[usize]) -> Vec<Duration> {
    sizes
        .iter()
        .copied()
        .map(|n| {
            let mut c = nalgebra::DMatrix::<T>::zeros(n, n);
            let a = nalgebra::DMatrix::<T>::zeros(n, n);
            let b = nalgebra::DMatrix::<T>::zeros(n, n);

            let time = timeit(|| {
                a.mul_to(&b, &mut c);
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
            let b = Mat::<T>::zeros(n, n);

            let time = timeit(|| {
                faer_core::mul::matmul(
                    c.as_mut(),
                    Conj::No,
                    a.as_ref(),
                    Conj::No,
                    b.as_ref(),
                    Conj::No,
                    None,
                    T::one(),
                    parallelism,
                );
            });

            let _ = c;

            time
        })
        .map(Duration::from_secs_f64)
        .collect()
}
