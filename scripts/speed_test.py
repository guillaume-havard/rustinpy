import time

import numpy as np

import rustinpy
import rustinpy.py

if __name__ == "__main__":
    np.random.seed(42)

    print("Add one")
    
    points = np.random.rand(50000, 10000)
    print("array shape:", points.shape)

    start_time = time.time()
    points = rustinpy.py.add_one(points)
    print(f"Execution time (Python): {time.time() - start_time:.3f} seconds")

    start_time = time.time()
    rustinpy.add_one(points)
    print(f"Execution time (rust): {time.time() - start_time:.3f} seconds")

    print("----------------------------------")

    print("Pairwise distances")

    points = np.random.rand(10000, 3)
    print("array shape:", points.shape)

    start_time = time.time()
    distances = rustinpy.py.pairwise_distances_raw(points)
    print(f"Execution time (Python): {time.time() - start_time:.3f} seconds")

    start_time = time.time()
    distances = rustinpy.py.pairwise_distances_np(points)
    print(f"Execution time (Numpy): {time.time() - start_time:.3f} seconds")
    
    distance_ref = distances

    start_time = time.time()
    distances = rustinpy.pairwise_distances_raw(points)
    print(f"Execution time (rust): {time.time() - start_time:.3f} seconds")

    if not np.allclose(distances, distance_ref):
        raise ValueError("Distances are not equal for rust")

    start_time = time.time()
    distances = rustinpy.pairwise_distances_broadcast(points)
    print(f"Execution time (rust ndarray): {time.time() - start_time:.3f} seconds")

    if not np.allclose(distances, distance_ref):
        raise ValueError("Distances are not equal for rust broadcast")

    start_time = time.time()
    distances = rustinpy.pairwise_distances_rayon(points)
    print(f"Execution time (rust rayon): {time.time() - start_time:.3f} seconds")

    if not np.allclose(distances, distance_ref):
        raise ValueError("Distances are not equal for rust rayon")

    start_time = time.time()
    distances = rustinpy.pairwise_distances_ndarray_parralel(points)
    print(f"Execution time (rust ndarray parralel): {time.time() - start_time:.3f} seconds")

    if not np.allclose(distances, distance_ref):
        raise ValueError("Distances are not equal for rust ndarray parralel")