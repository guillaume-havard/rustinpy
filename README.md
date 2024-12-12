# Rust in Python

This project demonstrates how to integrate Rust code into Python applications to improve
performance in computationally intensive tasks.

THis package is an hybrid one between python and rust

* `src/` contains rust sources
* `src_python/` contains python sources

## Performance Benchmark

Use ``scripts/speed_test.py``

The following benchmarks compare the execution times of equivalent operations performed using pure
Python, NumPy, and Rust with various libraries and techniques:

```
Add one
array shape: (50000, 10000)
Execution time (Python): 6.225 seconds
Execution time (rust): 0.324 seconds
----------------------------------
Pairwise distances
array shape: (10000, 3)
Execution time (Python): 382.692 seconds
Execution time (Numpy): 2.525 seconds
Execution time (rust): 6.646 seconds
Execution time (rust ndarray): 4.203 seconds
Execution time (rust rayon): 0.838 seconds
Execution time (rust ndarray parralel): 0.495 seconds
```