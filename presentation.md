# nalgebra

Quick note: `nalebra_test` folder is just for examples seen below.

Actual benchmarking now done in the `benchmark` folder which is forked from faer-rs.

## Why nalgebra?

## Group Project Experiment

- Perform in-depth analysis of the speed of performing certain operations (QR decomposition, multiplication, SVD) in nalgebra vs. Julia (built-ins or custom, could be interesting to see how a naive QR solution like in class would hold up).
- See which cases SIMD matters or helps with performance.
- Fairly easy to present in terms of visuals (graphically with time on x-axis and dimension of matrix on y-axis)
- Use plotting libraries to visualize the effect that smaller matrix dimensions have on certain operations between different libraries.
- Produce an analysis of results in order to explain why dynamic vs static allocation, and why smaller vs larger matrix dimensions lead to an improvement in speed.

## Questions

- Prof. Brown mentioned that this tool is popular for unbatched processing of small dimension matrices. Why is this?
- How can I perform a test between batched and unbatched processing of larger and smaller matrices?
- How would enabling SIMD impact performance of multiplying matrices? (this is pretty easy to answer with the [simba](https://docs.rs/simba/latest/simba/) crate)
- Does Rust have higher precision than Julia? Is this why cond(A) is different?
- Based on preliminary tests... why is Julia so much faster for larger dimension arrays?

## Methods of Interest
- Matrix multiplication
- Triangular solve
- LU decomposition w/ partial pivoting
- QR decomposition w/ no pivoting
- Square matrix SVD

## Other information

[Jump](#more-info) to the bottom

## Jump to benchmarks at the bottom (click [here](nalgebra_test/benches/matrix_multiply.rs) for code)

Ahead are just some examples of using the library, click [here](#benchmarks) to see benchmarks. Code for benchmarks is located in `nalgebra_test/benches/*.rs`.

To see historical plots of multiple passes through benchmarks, open `nalgebra_test/target/criterion/<benchmark name>/report/index.html`.

## Examples

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
let m1 = na::Matrix3::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
let v1 = na::Vector3::new(1.0, 2.0, 3.0);
let m2 = m1 * v1;
let m3 = m1 * m2;
println!("m1 * v1 = {}", m2);
println!("m1 * m2 = {}", m3);
// Output
m1 * v1 =
  ┌    ┐
  │ 14 │
  │ 32 │
  │ 50 │
  └    ┘
m1 * m2 =
  ┌     ┐
  │ 228 │
  │ 516 │
  │ 804 │
  └     ┘
```

## QR Decomposition

Resulting matrix after multiplying `q * r` has some rounding errors.

```rust
// Perform QR decomposition on a matrix
let m4 = na::Matrix3::new(1.0, 2.0, 3.0, 1.0, 3.0, 2.0, 3.0, 1.0, 2.0);
let qr = m4.qr();
let q = qr.q();
let r = qr.r();
println!("q = {}r = {}", q, r);
println!("qr = {}", q * r);
q =
  ┌                                                                ┐
  │  0.30151134457776374  0.44494920831460993   0.8432740427115676 │
  │  0.30151134457776363   0.7945521577046606  -0.5270462766947301 │
  │   0.9045340337332908 -0.41316712200642347 -0.10540925533894596 │
  └                                                                ┘
r =
  ┌                                                          ┐
  │    3.3166247903554  2.412090756622109    3.3166247903554 │
  │                  0  2.860387767736777  2.097617696340304 │
  │                  0                  0 1.2649110640673513 │
  └                                                          ┘
qr =
  ┌                                                          ┐
  │ 1.0000000000000004  2.000000000000001  3.000000000000001 │
  │                  1  3.000000000000001 2.0000000000000018 │
  │ 2.9999999999999996 0.9999999999999993 1.9999999999999987 │
  └                                                          ┘
```

## SVD

```rust
// Easy to calculate SVD of a matrix
let svd = m4.svd(true, true);
println!("svd = {}", svd.singular_values);
// Output:
svd =
  ┌                    ┐
  │ 6.0591880168285615 │
  │ 2.0960691446239803 │
  │ 0.9448463989858188 │
  └                    ┘
```

## Condition Number

```rust
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
let (norm_b, norm_b_i, cond_b) = cond(b); // I just took the above code and made is a more general function
println!("norm_b = {}", norm_b);
println!("norm_b_i = {}", norm_b_i);
println!("cond_b = {}", cond_b);
// Output
norm_m2x2 = 5.477225575051661
norm_inverse_m4x4 = 2.7386127875258306
cond = 15 // in Julia this is 14.933034373659268
norm_b = 3.152808272001328
norm_b_i = 105.4450927090744
cond_b = 332.4481605351167 // 332.4451525201234 in Julia
```

---

# Benchmarks

Benchmarks are completed using `criterion-rs` with no harness selected.

Here are some benchmarks for multiplying two matrices, creating two matrices (to see if this process is the bottleneck or if multiplying is), and creating + multiplying two matrices in one function.

All these tests collect 100 samples over a certain number of iterations. For lower dimensions, these iterations can easily reach over 1 million (whereas for 1024x1024, only a few hundred iterations are ran).

Comparison to benchmarks ran for faer-rs (another Rust linear algebra library): [here](https://faer-rs.github.io/bench-f64/) (both of these were done using f64 values)

---

### Multiplying matrix of dimension nxn

| n    | nalgebra  | julia |
| ---- | --------- | ----- |
| 16   | 309.58 ns |       |
| 32   | 1.6607 µs |       |
| 64   | 10.418 µs |       |
| 128  | 78.187 µs |       |
| 256  | 596.41 µs |       |
| 512  | 4.7759 ms |       |
| 1024 | 39.149 ms |       |

### Multiplying and creating matrix of dimension nxn

| n    | nalgebra  | julia      |
| ---- | --------- | ---------- |
| 16   | 651.31 ns |            |
| 32   | 2.3527 µs |            |
| 64   | 13.028 µs |            |
| 128  | 86.872 µs |            |
| 256  | 629.74 µs | 331.067 μs |
| 512  | 4.8887 ms |            |
| 1024 | 41.453 ms | 9.811 ms   |

### Creating matrix of dimension nxn

| n    | nalgebra  | julia |
| ---- | --------- | ----- |
| 16   | 258.73 ns |       |
| 32   | 713.40 ns |       |
| 64   | 2.4165 µs |       |
| 128  | 9.2018 µs |       |
| 256  | 36.022 µs |       |
| 512  | 145.50 µs |       |
| 1024 | 1.4949 ms |       |

# More info

This is a general purpose linear algebra library built around the Rust ecosystem. It offers a lot of support for 3D linear algebra and gaming/graphics.

This project is sponsored by three games studios:

- Fragcolor
- Embark Studios
- Resolution

This website does not offer any benchmarking so that is why I want to do it myself.
