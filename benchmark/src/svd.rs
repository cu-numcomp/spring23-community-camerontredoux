use super::timeit;
use crate::{random, types};
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
                    for j in 0..n {
                        c[(i, j)] = random();
                    }
                }

                let time = timeit(|| {
                    c.clone().svd(true, true);
                });

                time
            }
            2 => {
                let mut c = nalgebra::Matrix2::<T>::zeros();
                for i in 0..n {
                    for j in 0..n {
                        c[(i, j)] = random();
                    }
                }

                let time = timeit(|| {
                    c.clone().svd(true, true);
                });

                time
            }
            3 => {
                let mut c = nalgebra::Matrix3::<T>::zeros();
                for i in 0..n {
                    for j in 0..n {
                        c[(i, j)] = random();
                    }
                }

                let time = timeit(|| {
                    c.clone().svd(true, true);
                });

                time
            }
            4 => {
                let mut c = nalgebra::Matrix4::<T>::zeros();
                for i in 0..n {
                    for j in 0..n {
                        c[(i, j)] = random();
                    }
                }

                let time = timeit(|| {
                    c.clone().svd(true, true);
                });

                time
            }
            5 => {
                let mut c = nalgebra::Matrix5::<T>::zeros();
                for i in 0..n {
                    for j in 0..n {
                        c[(i, j)] = random();
                    }
                }

                let time = timeit(|| {
                    c.clone().svd(true, true);
                });

                time
            }
            6 => {
                let mut c = nalgebra::Matrix6::<T>::zeros();
                for i in 0..n {
                    for j in 0..n {
                        c[(i, j)] = random();
                    }
                }

                let time = timeit(|| {
                    c.clone().svd(true, true);
                });

                time
            }
            7 => {
                let mut c = types::Matrix7::<T>::zeros();
                for i in 0..n {
                    for j in 0..n {
                        c[(i, j)] = random();
                    }
                }

                let time = timeit(|| {
                    c.clone().svd(true, true);
                });

                time
            }
            8 => {
                let mut c = types::Matrix8::<T>::zeros();
                for i in 0..n {
                    for j in 0..n {
                        c[(i, j)] = random();
                    }
                }

                let time = timeit(|| {
                    c.clone().svd(true, true);
                });

                time
            }
            9 => {
                let mut c = types::Matrix9::<T>::zeros();
                for i in 0..n {
                    for j in 0..n {
                        c[(i, j)] = random();
                    }
                }

                let time = timeit(|| {
                    c.clone().svd(true, true);
                });

                time
            }
            10 => {
                let mut c = types::Matrix10::<T>::zeros();
                for i in 0..n {
                    for j in 0..n {
                        c[(i, j)] = random();
                    }
                }

                let time = timeit(|| {
                    c.clone().svd(true, true);
                });

                time
            }
            11 => {
                let mut c = types::Matrix11::<T>::zeros();
                for i in 0..n {
                    for j in 0..n {
                        c[(i, j)] = random();
                    }
                }

                let time = timeit(|| {
                    c.clone().svd(true, true);
                });

                time
            }
            12 => {
                let mut c = types::Matrix12::<T>::zeros();
                for i in 0..n {
                    for j in 0..n {
                        c[(i, j)] = random();
                    }
                }

                let time = timeit(|| {
                    c.clone().svd(true, true);
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
                for j in 0..n {
                    c[(i, j)] = random();
                }
            }

            let time = timeit(|| {
                c.clone().svd(true, true);
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
                for j in 0..n {
                    c[(i, j)] = random();
                }
            }
            let mut s = Mat::<T>::zeros(n, n);
            let mut u = Mat::<T>::zeros(n, n);
            let mut v = Mat::<T>::zeros(n, n);

            let mut mem = GlobalMemBuffer::new(
                faer_svd::compute_svd_req::<T>(
                    n,
                    n,
                    faer_svd::ComputeVectors::Full,
                    faer_svd::ComputeVectors::Full,
                    parallelism,
                    faer_svd::SvdParams::default(),
                )
                .unwrap(),
            );
            let mut stack = DynStack::new(&mut mem);

            let time = timeit(|| {
                faer_svd::compute_svd(
                    c.as_ref(),
                    s.as_mut().diagonal(),
                    Some(u.as_mut()),
                    Some(v.as_mut()),
                    crate::epsilon::<T>(),
                    crate::min_positive::<T>(),
                    parallelism,
                    stack.rb_mut(),
                    faer_svd::SvdParams::default(),
                );
            });

            let _ = c;

            time
        })
        .map(Duration::from_secs_f64)
        .collect()
}
