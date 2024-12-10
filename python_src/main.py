import time

import numpy as np
import rustinpy


def pairwise_distances_raw(points):
    n = points.shape[0]
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distances[i, j] = np.sqrt(np.sum((points[i] - points[j]) ** 2))
    return distances

def pairwise_distances_np(points):
    squared_diff = np.sum((points[:, np.newaxis, :] - points[np.newaxis, :, :]) ** 2, axis=-1)
    return np.sqrt(squared_diff)

def add_one(points):
    return points + 1

if __name__ == "__main__":
    np.random.seed(42)

    print("Add one")
    
    points = np.random.rand(50000, 10000)
    print("array shape:", points.shape)

    start_time = time.time()
    points = add_one(points)
    print(f"Execution time (Python): {time.time() - start_time:.3f} seconds")

    start_time = time.time()
    rustinpy.add_one(points)
    print(f"Execution time (rust): {time.time() - start_time:.3f} seconds")

    print("----------------------------------")
    print("Pairwise distances")
    

    points = np.random.rand(10000, 3)
    print("array shape:", points.shape)

    start_time = time.time()
    distances = pairwise_distances_raw(points)
    print(f"Execution time (Python): {time.time() - start_time:.3f} seconds")

    start_time = time.time()
    distances = pairwise_distances_np(points)
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