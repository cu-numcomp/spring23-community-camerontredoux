extern crate nalgebra as na;

// Defining a custom column vector, statically sized, with 10 rows
type Vector10 = na::SVector<i32, 10>;

fn cond(a: na::Matrix2<f64>) -> (f64, f64, f64) {
    let a_i = a.try_inverse().unwrap();
    let norm_a: f64 = a.norm();
    let norm_a_i: f64 = a_i.norm();
    let cond: f64 = norm_a * norm_a_i;
    (norm_a, norm_a_i, cond)
}

fn main() {
    // Simple vectors and showing basic arithmetic.
    let vec1 = na::Vector2::new(1.0, 3.0);
    let vec2 = na::Vector2::new(1.0, 2.0);
    let vec10 = Vector10::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 0]);
    let vec_add = vec1 + vec2;
    // let vec_mul = vec1 * vec2;
    println!("{} + {} = {}", vec1, vec2, vec_add);
    // println!("{} * {} = {}", vec1, vec2, vec_mul);
    println!("vec10 = {}", vec10);

    // Create the identity matrix and a zeroes matrix.
    // Matrix4 just means a 4x4 matrix
    let id = na::Matrix4::<f32>::identity();
    let zero = na::Matrix4::<f32>::zeros();
    // Docs here are incorrect, need to specify type like below
    let d_zero = na::DMatrix::<i32>::identity(4, 3);
    println!("id = {}", id);
    println!("zero = {}", zero);
    println!("d_zero = {}", d_zero);

    // Can create a matrix using a closure.
    let matrix_fn = na::Matrix2x3::from_fn(|r, c| (r + 1) as f32 + (c + 1) as f32 / 10.0);
    println!("matrix_fn = {}", matrix_fn);

    // Use views to get a slice of a matrix without allocating more memory (just a pointer).
    let matrix_view = na::Matrix3::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    println!("row_1 = {}", matrix_view.row(0));
    println!("row_2 = {}", matrix_view.row(1));
    println!("col_1 = {}", matrix_view.column(0));
    println!("col_3 = {}", matrix_view.column(2));

    // Can also get a mutable view of a matrix.
    let mut matrix_view_mut = matrix_view.clone();
    matrix_view_mut.row_mut(0).fill(0.0);
    println!("matrix_view_mut = {}", matrix_view_mut);

    let m1 = na::Matrix3::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    let v1 = na::Vector3::new(1.0, 2.0, 3.0);
    let m2 = m1 * v1;
    let m3 = m1 * m2;
    println!("m1 * v1 = {}", m2);
    println!("m1 * m2 = {}", m3);

    // Perform QR decomposition on a matrix
    let m4 = na::Matrix3::new(1.0, 2.0, 3.0, 1.0, 3.0, 2.0, 3.0, 1.0, 2.0);
    let qr = m4.qr();
    let q = qr.q();
    let r = qr.r();
    println!("q = {}r = {}", q, r);
    println!("qr = {}", q * r);

    // Easy to calculate SVD of a matrix
    let svd = m4.svd(true, true);
    println!("svd = {}", svd.singular_values);

    // Condition number of a matrix is the norm of the matrix times the norm of the inverse
    let m2x2 = na::Matrix2::new(1.0, 2.0, 3.0, 4.0);
    let m2x2_i = m2x2.try_inverse().unwrap();
    // Get the norm of matrix4x4 (in this case its the L2 norm)
    let norm_m2x2: f64 = m2x2.norm();
    println!("norm_m2x2 = {}", norm_m2x2);
    // Get the norm of the inverse of matrix4x4
    let norm_inverse_m2x2: f64 = m2x2_i.norm();
    println!("norm_inverse_m4x4 = {}", norm_inverse_m2x2);
    let cond2x2: f64 = norm_m2x2 * norm_inverse_m2x2;
    println!("cond = {}", cond2x2);

    let b = na::Matrix2::new(1.0, 0.99, 1.99, 2.0);
    let (norm_b, norm_b_i, cond_b) = cond(b);
    println!("norm_b = {}", norm_b);
    println!("norm_b_i = {}", norm_b_i);
    println!("cond_b = {}", cond_b);
}
