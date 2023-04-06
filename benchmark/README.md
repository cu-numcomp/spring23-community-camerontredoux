# Benchmarking

## Credit

Note that most of this code is pulled from https://github.com/sarah-ek/faer-rs/tree/main/faer-bench

Just want to point that out. What I (Cameron) did was modify this to run on matrices of size 1 to 12 and included an additional benchmark for statically allocated matrices in nalgebra.

All other credit goes to sarah-ek.

## Libraries

- eigen is a C++ linear algebra library
- ndarray-linalg is a Rust linear algebra library
- faer-rs is another Rust linear algebra library
  - faer also supports parallelism with `rayon` so that will show up in the results
- nalgebra is the Rust linear algebra library we are experimenting with
  - nalgebra_d in the results is the dynamically allocated matrix version
  - nalgebra_s is the staticaly allocated matrix

## Results
