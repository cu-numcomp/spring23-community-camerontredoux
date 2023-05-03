use coe::is_same;
use eyre::Result;
use faer_core::{c32, c64, Parallelism};
use human_repr::HumanDuration;
use std::{
    fs::{self, File},
    io::Write,
    time::Duration,
};

extern crate blas_src;
extern crate openmp_sys;

fn random<T: 'static>() -> T {
    if is_same::<f32, T>() {
        coe::coerce_static(rand::random::<f32>())
    } else if is_same::<f64, T>() {
        coe::coerce_static(rand::random::<f64>())
    } else if is_same::<c32, T>() {
        coe::coerce_static(c32::new(rand::random(), rand::random()))
    } else if is_same::<c64, T>() {
        coe::coerce_static(c64::new(rand::random(), rand::random()))
    } else {
        unimplemented!()
    }
}

fn epsilon<T: faer_core::ComplexField>() -> T::Real {
    if is_same::<f32, T>() {
        coe::coerce_static(f32::EPSILON)
    } else if is_same::<f64, T>() {
        coe::coerce_static(f64::EPSILON)
    } else if is_same::<c32, T>() {
        coe::coerce_static(f32::EPSILON)
    } else if is_same::<c64, T>() {
        coe::coerce_static(f64::EPSILON)
    } else {
        unimplemented!()
    }
}

fn min_positive<T: faer_core::ComplexField>() -> T::Real {
    if is_same::<f32, T>() {
        coe::coerce_static(f32::MIN_POSITIVE)
    } else if is_same::<f64, T>() {
        coe::coerce_static(f64::MIN_POSITIVE)
    } else if is_same::<c32, T>() {
        coe::coerce_static(f32::MIN_POSITIVE)
    } else if is_same::<c64, T>() {
        coe::coerce_static(f64::MIN_POSITIVE)
    } else {
        unimplemented!()
    }
}

fn time(mut f: impl FnMut()) -> f64 {
    let instant = std::time::Instant::now();
    f();
    instant.elapsed().as_secs_f64()
}

fn timeit(f: impl FnMut()) -> f64 {
    let mut f = f;
    let min = 1e-0;
    let once = time(&mut f);
    if once > min {
        return once;
    }

    let ten = time(|| {
        for _ in 0..10 {
            f()
        }
    });

    if ten > min {
        return ten / 10.0;
    }

    let n = (min * 10.0 / ten).ceil() as u64;
    time(|| {
        for _ in 0..n {
            f()
        }
    }) / n as f64
}

mod cholesky;
mod gemm;
mod no_piv_qr;
mod partial_piv_lu;
mod svd;
mod trsm;
mod types;

macro_rules! printwriteln {
    ($out: expr, $($arg:tt)*) => {
        {
            println!($($arg)*);
            writeln!($out, $($arg)*)
        }
    };
}

fn print_results(
    output: &mut dyn Write,
    input_sizes: &[usize],
    faer: &[Duration],
    nalgebra_d: &[Duration],
    nalgebra_s: Option<&[Duration]>,
    eigen: &[Duration],
) -> Result<()> {
    let fmt = |d: Duration| {
        if d == Duration::ZERO {
            "-".to_string()
        } else {
            format!("{}", d.human_duration())
        }
    };
    printwriteln!(
        output,
        "{:>5} {:>10} {:>10} {:>10} {:>10}",
        "n",
        "faer",
        "nalgebra_d",
        "nalgebra_s",
        "eigen",
    )?;

    for (i, n) in input_sizes.iter().copied().enumerate() {
        printwriteln!(
            output,
            "{:5} {:>10} {:>10} {:>10} {:>10}",
            n,
            fmt(faer[i]),
            fmt(nalgebra_d[i]),
            fmt(nalgebra_s.map(|v| v[i]).unwrap_or(Duration::ZERO)),
            fmt(eigen[i]),
        )?;
    }
    Ok(())
}

mod eigen {
    extern "C" {
        pub fn gemm_f64(out: *mut f64, inputs: *const usize, count: usize);
        pub fn trsm_f64(out: *mut f64, inputs: *const usize, count: usize);
        pub fn chol_f64(out: *mut f64, inputs: *const usize, count: usize);
        pub fn plu_f64(out: *mut f64, inputs: *const usize, count: usize);
        pub fn qr_f64(out: *mut f64, inputs: *const usize, count: usize);
        pub fn svd_f64(out: *mut f64, inputs: *const usize, count: usize);

        pub fn gemm_c64(out: *mut f64, inputs: *const usize, count: usize);
        pub fn trsm_c64(out: *mut f64, inputs: *const usize, count: usize);
        pub fn chol_c64(out: *mut f64, inputs: *const usize, count: usize);
        pub fn plu_c64(out: *mut f64, inputs: *const usize, count: usize);
        pub fn qr_c64(out: *mut f64, inputs: *const usize, count: usize);
        pub fn svd_c64(out: *mut f64, inputs: *const usize, count: usize);
    }
}

fn eigen(
    f: unsafe extern "C" fn(*mut f64, *const usize, usize),
    input_sizes: &[usize],
) -> Vec<Duration> {
    let count = input_sizes.len();
    let mut v = vec![0.0; count];
    unsafe {
        f(v.as_mut_ptr(), input_sizes.as_ptr(), count);
    }
    v.into_iter().map(Duration::from_secs_f64).collect()
}

fn main() -> Result<()> {
    // let input_sizes = vec![32, 64, 96, 128, 192, 256, 384, 512, 640, 768, 896, 1024];
    // fs::create_dir("data_large").ok();
    let input_sizes = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 64, 128, 256, 1024];
    fs::create_dir("data").ok();

    {
        println!("f64");
        let mut file = File::create("data/f64.md")?;
        printwriteln!(
            file,
            "
## Matrix multiplication

Multiplication of two square matrices of dimension `n`.

```"
        )?;
        print_results(
            &mut file,
            &input_sizes,
            &gemm::faer::<f64>(&input_sizes, Parallelism::None),
            &gemm::nalgebra_d::<f64>(&input_sizes),
            Some(&gemm::nalgebra_s::<f64>(&input_sizes)),
            &eigen(eigen::gemm_f64, &input_sizes),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(file,"
## Triangular solve

Solving `AX = B` in place where `A` and `B` are two square matrices of dimension `n`, and `A` is a triangular matrix.

```")?;
        print_results(
            &mut file,
            &input_sizes,
            &trsm::faer::<f64>(&input_sizes, Parallelism::None),
            &trsm::nalgebra::<f64>(&input_sizes),
            Some(&trsm::nalgebra_s::<f64>(&input_sizes)),
            &eigen(eigen::trsm_f64, &input_sizes),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(
            file,
            "
## Cholesky decomposition

Factorizing a square matrix with dimension `n` as `L×L.T`, where `L` is lower triangular.

```"
        )?;
        print_results(
            &mut file,
            &input_sizes,
            &cholesky::faer::<f64>(&input_sizes, Parallelism::None),
            &cholesky::nalgebra::<f64>(&input_sizes),
            Some(&cholesky::nalgebra_s::<f64>(&input_sizes)),
            &eigen(eigen::chol_f64, &input_sizes),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(file,"
## LU decomposition with partial pivoting

Factorizing a square matrix with dimension `n` as `P×L×U`, where `P` is a permutation matrix, `L` is unit lower triangular and `U` is upper triangular.

```")?;
        print_results(
            &mut file,
            &input_sizes,
            &partial_piv_lu::faer::<f64>(&input_sizes, Parallelism::None),
            &partial_piv_lu::nalgebra::<f64>(&input_sizes),
            Some(&partial_piv_lu::nalgebra_s::<f64>(&input_sizes)),
            &eigen(eigen::plu_f64, &input_sizes),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(file,"
## QR decomposition with no pivoting

Factorizing a square matrix with dimension `n` as `QR`, where `Q` is unitary and `R` is upper triangular.

```")?;
        print_results(
            &mut file,
            &input_sizes,
            &no_piv_qr::faer::<f64>(&input_sizes, Parallelism::None),
            &no_piv_qr::nalgebra::<f64>(&input_sizes),
            Some(&no_piv_qr::nalgebra_s::<f64>(&input_sizes)),
            &eigen(eigen::qr_f64, &input_sizes),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(
            file,
            "
## Square matrix singular value decomposition

Computing the SVD of a square matrix with dimension `n`.

```"
        )?;
        print_results(
            &mut file,
            &input_sizes,
            &svd::faer::<f64>(&input_sizes, Parallelism::None),
            &svd::nalgebra::<f64>(&input_sizes),
            Some(&svd::nalgebra_s::<f64>(&input_sizes)),
            &eigen(eigen::svd_f64, &input_sizes),
        )?;
        printwriteln!(file, "```")?;
    }
    {
        println!("c64 - complex f64");
        let mut file = File::create("data/c64.md")?;
        printwriteln!(
            file,
            "
## Matrix multiplication

Multiplication of two square matrices of dimension `n`.

```"
        )?;
        print_results(
            &mut file,
            &input_sizes,
            &gemm::faer::<c64>(&input_sizes, Parallelism::None),
            &gemm::nalgebra_d::<c64>(&input_sizes),
            Some(&gemm::nalgebra_s::<c64>(&input_sizes)),
            &eigen(eigen::gemm_c64, &input_sizes),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(file,"
## Triangular solve

Solving `AX = B` in place where `A` and `B` are two square matrices of dimension `n`, and `A` is a triangular matrix.

```")?;
        print_results(
            &mut file,
            &input_sizes,
            &trsm::faer::<c64>(&input_sizes, Parallelism::None),
            &trsm::nalgebra::<c64>(&input_sizes),
            Some(&trsm::nalgebra_s::<c64>(&input_sizes)),
            &eigen(eigen::trsm_c64, &input_sizes),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(
            file,
            "
## Cholesky decomposition

Factorizing a square matrix with dimension `n` as `L×L.T`, where `L` is lower triangular.

```"
        )?;
        print_results(
            &mut file,
            &input_sizes,
            &cholesky::faer::<c64>(&input_sizes, Parallelism::None),
            &cholesky::nalgebra::<c64>(&input_sizes),
            Some(&cholesky::nalgebra_s::<c64>(&input_sizes)),
            &eigen(eigen::chol_c64, &input_sizes),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(file,"
## LU decomposition with partial pivoting

Factorizing a square matrix with dimension `n` as `P×L×U`, where `P` is a permutation matrix, `L` is unit lower triangular and `U` is upper triangular.

```")?;
        print_results(
            &mut file,
            &input_sizes,
            &partial_piv_lu::faer::<c64>(&input_sizes, Parallelism::None),
            &partial_piv_lu::nalgebra::<c64>(&input_sizes),
            Some(&partial_piv_lu::nalgebra_s::<c64>(&input_sizes)),
            &eigen(eigen::plu_c64, &input_sizes),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(file,"
## QR decomposition with no pivoting

Factorizing a square matrix with dimension `n` as `QR`, where `Q` is unitary and `R` is upper triangular.

```")?;
        print_results(
            &mut file,
            &input_sizes,
            &no_piv_qr::faer::<c64>(&input_sizes, Parallelism::None),
            &no_piv_qr::nalgebra::<c64>(&input_sizes),
            Some(&no_piv_qr::nalgebra_s::<c64>(&input_sizes)),
            &eigen(eigen::qr_c64, &input_sizes),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(
            file,
            "
## Square matrix singular value decomposition

Computing the SVD of a square matrix with dimension `n`.

```"
        )?;
        print_results(
            &mut file,
            &input_sizes,
            &svd::faer::<c64>(&input_sizes, Parallelism::None),
            &svd::nalgebra::<c64>(&input_sizes),
            Some(&svd::nalgebra_s::<c64>(&input_sizes)),
            &eigen(eigen::svd_c64, &input_sizes),
        )?;
        printwriteln!(file, "```")?;
    }
    Ok(())
}
