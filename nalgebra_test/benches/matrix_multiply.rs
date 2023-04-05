use criterion::{black_box, criterion_group, criterion_main, Criterion};

extern crate nalgebra as na;

fn matrix_create_mult(n: u64) -> na::DMatrix<f64> {
    let (m1, m2) = create_matrix(n);

    m1 * m2
}

fn matrix_mult2(m1: &na::DMatrix<f64>, m2: &na::DMatrix<f64>) -> na::DMatrix<f64> {
    m1 * m2
}

fn matrix_create(n: u64) {
    let (_, _) = create_matrix(n);
}

fn bench_mult_matrix(c: &mut Criterion) {
    let (m1, m2) = create_matrix(16);
    c.bench_function("matrix_mult 16x16", |b| {
        b.iter(|| matrix_mult2(black_box(&m1), black_box(&m2)))
    });
    let (m1, m2) = create_matrix(32);
    c.bench_function("matrix_mult 32x32", |b| {
        b.iter(|| matrix_mult2(black_box(&m1), black_box(&m2)))
    });
    let (m1, m2) = create_matrix(64);
    c.bench_function("matrix_mult 64x64", |b| {
        b.iter(|| matrix_mult2(black_box(&m1), black_box(&m2)))
    });
    let (m1, m2) = create_matrix(128);
    c.bench_function("matrix_mult 128x128", |b| {
        b.iter(|| matrix_mult2(black_box(&m1), black_box(&m2)))
    });
    let (m1, m2) = create_matrix(256);
    c.bench_function("matrix_mult 256x256", |b| {
        b.iter(|| matrix_mult2(black_box(&m1), black_box(&m2)))
    });
    let (m1, m2) = create_matrix(512);
    c.bench_function("matrix_mult 512x512", |b| {
        b.iter(|| matrix_mult2(black_box(&m1), black_box(&m2)))
    });
    let (m1, m2) = create_matrix(1024);
    c.bench_function("matrix_mult 1024x1024", |b| {
        b.iter(|| matrix_mult2(black_box(&m1), black_box(&m2)))
    });
}

fn bench_create_and_mult_matrix(c: &mut Criterion) {
    c.bench_function("matrix_create_mult 16x16", |b| {
        b.iter(|| matrix_create_mult(black_box(16)))
    });
    c.bench_function("matrix_create_mult 32x32", |b| {
        b.iter(|| matrix_create_mult(black_box(32)))
    });
    c.bench_function("matrix_create_mult 64x64", |b| {
        b.iter(|| matrix_create_mult(black_box(64)))
    });
    c.bench_function("matrix_create_mult 128x128", |b| {
        b.iter(|| matrix_create_mult(black_box(128)))
    });
    c.bench_function("matrix_create_mult 256x256", |b| {
        b.iter(|| matrix_create_mult(black_box(256)))
    });
    c.bench_function("matrix_create_mult 512x512", |b| {
        b.iter(|| matrix_create_mult(black_box(512)))
    });
    c.bench_function("matrix_create_mult 1024x1024", |b| {
        b.iter(|| matrix_create_mult(black_box(1024)))
    });
}

fn bench_create_matrix(c: &mut Criterion) {
    c.bench_function("matrix_create 16x16", |b| {
        b.iter(|| matrix_create(black_box(16)))
    });
    c.bench_function("matrix_create 32x32", |b| {
        b.iter(|| matrix_create(black_box(32)))
    });
    c.bench_function("matrix_create 64x64", |b| {
        b.iter(|| matrix_create(black_box(64)))
    });
    c.bench_function("matrix_create 128x128", |b| {
        b.iter(|| matrix_create(black_box(128)))
    });
    c.bench_function("matrix_create 256x256", |b| {
        b.iter(|| matrix_create(black_box(256)))
    });
    c.bench_function("matrix_create 512x512", |b| {
        b.iter(|| matrix_create(black_box(512)))
    });
    c.bench_function("matrix_create 1024x1024", |b| {
        b.iter(|| matrix_create(black_box(1024)))
    });
}

fn create_matrix(n: u64) -> (na::DMatrix<f64>, na::DMatrix<f64>) {
    let m1 = na::DMatrix::<f64>::from_fn(n as usize, n as usize, |r, c| {
        (r + 1) as f64 + (c + 1) as f64 / 10.0
    });

    let m2 = na::DMatrix::<f64>::from_fn(n as usize, n as usize, |r, c| {
        (r + 1) as f64 + (c + 1) as f64 / 10.0
    });

    (m1, m2)
}

criterion_group!(
    benches,
    bench_mult_matrix,
    bench_create_and_mult_matrix,
    bench_create_matrix
);
criterion_main!(benches);
