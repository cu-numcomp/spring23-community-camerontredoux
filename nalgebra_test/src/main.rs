extern crate nalgebra as na;

// Defining a custom column vector, statically sized, with 10 rows
type Vector10 = na::SVector<i32, 10>;

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
}
