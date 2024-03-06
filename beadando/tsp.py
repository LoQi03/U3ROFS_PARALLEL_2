import math
import random
import time
import pyopencl as cl
import numpy as np


def read_tsp_file(filename):
    tsp = []
    with open(filename, 'r') as file:
        start_reading = False
        for line in file:
            if line.strip() == "NODE_COORD_SECTION":
                start_reading = True
                continue
            if start_reading and line.strip() != "EOF":
                index, x, y = map(float, line.split())
                tsp.append((x, y))
    return tsp


def distance(c1, c2):
    return math.sqrt((c2[0] - c1[0]) ** 2 + (c2[1] - c1[1]) ** 2)


def local_search(tsp, initial_order):
    tsp_size = len(tsp)
    order_size = len(initial_order)
    distances = np.zeros((tsp_size, tsp_size))

    for i in range(tsp_size):
        for j in range(tsp_size):
            index1 = initial_order[i]
            index2 = initial_order[j]
            dx = tsp[index1][0] - tsp[index2][0]
            dy = tsp[index1][1] - tsp[index2][1]
            distances[i][j] = np.sqrt(dx*dx + dy*dy)

    visited = np.zeros(tsp_size, dtype=int)
    current_city = 0
    visited[current_city] = 1
    new_order = [current_city]

    total_dist = 0.0

    for _ in range(1, tsp_size):
        nearest_city = -1
        shortest_dist = 1e9

        for j in range(tsp_size):
            if not visited[j] and distances[current_city][j] < shortest_dist:
                shortest_dist = distances[current_city][j]
                nearest_city = j

        visited[nearest_city] = 1
        new_order.append(nearest_city)
        current_city = nearest_city
        total_dist += shortest_dist

    return new_order, total_dist


def run_tsp_no_kernel(tsp, initial_order):
    start_time = time.time()
    best_order, total_distance = local_search(tsp, initial_order)

    print("Best order:", best_order)
    print("Total distance:", total_distance)

    end_time = time.time()
    duration = (end_time - start_time) * 1000  # milliseconds
    print("Execution time:", duration, "milliseconds")


kernel_code = """
__kernel void local_search(__global const double* tsp, const int tsp_size, __global const int* initial_order, const int order_size, __global int* new_order, __global double* best_dist) {
    int cityIndex = get_global_id(0);

    // Distance matrix
    double distances[150][2];
    for (int i = 0; i < 150; ++i) {
        for (int j = 0; j < 2; ++j) {
            distances[i][j] = 0; // initialize distances
        }
    }


    for (int i = 0; i < 150; ++i) {
        for (int j = 0; j < 2; ++j) {
            int index1 = initial_order[i];
            int index2 = initial_order[j];
            double dx = tsp[index1 * 3] - tsp[index2 * 3];
            double dy = tsp[index1 * 3 + 1] - tsp[index2 * 3 + 1];
            distances[i][j] = sqrt(dx*dx + dy*dy); // Euclidean distance
        }
    }

    int visited[150];
    for (int i = 0; i < 150; ++i) {
        visited[i] = 0;
    }

    int currentCity = cityIndex;
    visited[currentCity] = 1;
    new_order[0] = currentCity;

    double totalDist = 0.0;

    for (int i = 1; i < 150; ++i) {
        int nearestCity = -1;
        double shortestDist = 1e9;

        for (int j = 0; j < 150; ++j) {
            if (!visited[j] && distances[currentCity][j] < shortestDist) {
                shortestDist = distances[currentCity][j];
                nearestCity = j;
            }
        }

        visited[nearestCity] = 1;
        new_order[i] = nearestCity;
        currentCity = nearestCity;
        totalDist += shortestDist;
    }

    if (totalDist < *best_dist) {
        *best_dist = totalDist;
    }
}
"""


def run_tsp_on_kernel(tsp, initial_order):
    # Create OpenCL context and queue
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    context = cl.Context([device])
    queue = cl.CommandQueue(context)
    # Compile the kernel
    program = cl.Program(context, kernel_code).build()

    # Convert tsp and initial_order to numpy arrays
    tsp_np = np.array(tsp, dtype=np.double)
    initial_order_np = np.array(initial_order, dtype=np.int32)
    # Since each city has 3 coordinates (x, y, z)
    tsp_size = tsp_np.shape[0] // 3
    order_size = initial_order_np.shape[0]

    # Create OpenCL buffers
    tsp_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY |
                           cl.mem_flags.COPY_HOST_PTR, hostbuf=tsp_np)
    initial_order_buffer = cl.Buffer(
        context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=initial_order_np)
    new_order_buffer = cl.Buffer(
        context, cl.mem_flags.WRITE_ONLY, initial_order_np.nbytes)
    best_dist_buffer = cl.Buffer(context, cl.mem_flags.READ_WRITE |
                                 cl.mem_flags.COPY_HOST_PTR, hostbuf=np.array([np.inf], dtype=np.double))
    start_time = time.time()
    # Execute the kernel
    program.local_search(queue, (tsp_size,), None, tsp_buffer, np.int32(tsp_size), initial_order_buffer,
                         np.int32(order_size), new_order_buffer, best_dist_buffer)
    queue.finish()

    # Retrieve the results
    new_order = np.empty_like(initial_order_np)
    best_dist = np.empty(1, dtype=np.double)
    cl.enqueue_copy(queue, new_order, new_order_buffer).wait()
    cl.enqueue_copy(queue, best_dist, best_dist_buffer).wait()

    print("Best order:", list(new_order))
    print("Total distance:", best_dist[0])

    end_time = time.time()
    duration = (end_time - start_time) * 1000  # milliseconds
    print("Execution time:", duration, "milliseconds")


if __name__ == "__main__":
    print("Reading tsp file...")
    tsp = read_tsp_file("src/tsp_problems/xqg237.tsp")

    initial_order = list(range(len(tsp)))
    print("Running TSP(no kernel)")
    run_tsp_no_kernel(tsp, initial_order)
    print("\nRunning TSP(on kernel)")
    run_tsp_on_kernel(tsp, initial_order)
