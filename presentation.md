# nalgebra

## Need to know

nalgebra uses column vectors so it looks slightly different than the array representation of vectors in Julia. This is console output of the vectors:

```
  ┌   ┐   ┌   ┐
  │ 1 │   │ 1 │
  │ 2 │ + │ 3 │
  └   ┘   └   ┘
 =
  ┌   ┐
  │ 2 │
  │ 5 │
  └   ┘

  ┌   ┐
  │ 1 │
  │ 2 │
  │ 3 │
  │ 4 │
  │ 5 │
  │ 6 │
  │ 7 │
  │ 8 │
  │ 9 │
  │ 0 │
  └   ┘

```

Ways to create vectors/matrices and some cool tools:

```rust
// Create the identity matrix and a zeroes matrix.
// Matrix4 just means a 4x4 matrix
let id = na::Matrix4::<f32>::identity();
let zero = na::Matrix4::<f32>::zeros();
println!("id = {}", id);
println!("zero = {}", zero);
// Output:
id =
  ┌         ┐
  │ 1 0 0 0 │
  │ 0 1 0 0 │
  │ 0 0 1 0 │
  │ 0 0 0 1 │
  └         ┘
zero =
  ┌         ┐
  │ 0 0 0 0 │
  │ 0 0 0 0 │
  │ 0 0 0 0 │
  │ 0 0 0 0 │
  └         ┘


// Can create a matrix using a closure.
let matrix_fn = na::Matrix2x3::from_fn(|r, c| (r + 1) as f32 + (c + 1) as f32 / 10.0);
println!("matrix_fn = {}", matrix_fn);
// Output:
matrix_fn =
  ┌             ┐
  │ 1.1 1.2 1.3 │
  │ 2.1 2.2 2.3 │
  └             ┘


// Use views to get a slice of a matrix without allocating more memory (just a pointer).
let matrix_view = na::Matrix3::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
println!("row_1 = {}", matrix_view.row(0));
println!("row_2 = {}", matrix_view.row(1));
println!("col_1 = {}", matrix_view.column(0));
println!("col_3 = {}", matrix_view.column(2));
//Output:
row_1 =
  ┌       ┐
  │ 1 2 3 │
  └       ┘
row_2 =
  ┌       ┐
  │ 4 5 6 │
  └       ┘
col_1 =
  ┌   ┐
  │ 1 │
  │ 4 │
  │ 7 │
  └   ┘
col_3 =
  ┌   ┐
  │ 3 │
  │ 6 │
  │ 9 │
  └   ┘
```

nalgebra has operator overloading, so you can easily perform arithmetic on matrices:

```rust
let m1 =
```
