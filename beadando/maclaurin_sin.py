import math
import time
import multiprocessing as mp
import pyopencl as cl
import numpy as np
import os
import matplotlib.pyplot as plt

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'


def load_kernel_code(filename):
    with open(filename, 'r') as f:
        return f.read()


kernel_code = load_kernel_code('src/kernels/maclaurin_sin.cl')


def run_opencl_kernel(kernel_code, kernel_name, x, start, end):
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    context = cl.Context([device])
    queue = cl.CommandQueue(
        context, properties=cl.command_queue_properties.PROFILING_ENABLE)
    program = cl.Program(context, kernel_code).build()

    result_size = end - start
    result = np.zeros(result_size, dtype=np.float64)
    result_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, result.nbytes)

    kernel = getattr(program, kernel_name)
    event = kernel(queue, (result_size,), None, result_buffer,
                   np.float64(x), np.int32(start), np.int32(end))

    cl.enqueue_copy(queue, result, result_buffer).wait()

    start_time = event.get_profiling_info(cl.profiling_info.START)
    end_time = event.get_profiling_info(cl.profiling_info.END)
    execution_time = end_time - start_time

    return np.sum(result), execution_time


def opencl_sin_maclaurin(x, n, num_intervals):
    interval_size = n // num_intervals
    intervals = [(i * interval_size, min((i + 1) * interval_size, n))
                 for i in range(num_intervals)]

    with mp.Pool(processes=num_intervals) as pool:
        results = pool.starmap(run_opencl_kernel, [(
            kernel_code, "maclaurin_sin", x, start, end) for start, end in intervals])

    sin_approx = sum(result[0] for result in results)
    max_execution_time = max(result[1] for result in results)

    print(f"sin({x}) = {sin_approx}\t idő: {max_execution_time / 1e6} s")

    return sin_approx, max_execution_time / 1e6


def factorial(n):
    result = 1.0
    for i in range(2, n + 1):
        result *= i
    return result


def sin_maclaurin_parallel(x, n, thread_count):
    start = time.time()
    result = x
    with mp.Pool(processes=thread_count) as pool:
        terms = pool.starmap(compute_sin_term, [
                             (x, i) for i in range(1, n + 1)])
        for term in terms:
            result += term
    print(f"sin({x}) = {result}\t ido: {time.time() - start}")
    return result, time.time() - start


def compute_sin_term(x, i):
    return math.pow(-1, i) * math.pow(x, 2 * i + 1) / factorial(2 * i + 1)


def seq_sin_maclaurin(x, n):
    start = time.time()
    result = x
    for i in range(1, n + 1):
        term = math.pow(-1, i) * math.pow(x, 2 * i + 1) / factorial(2 * i + 1)
        result += term
    print(f"sin({x}) = {result}\t ido: {time.time() - start}")
    return result, time.time() - start


def main():
    x = 1.0
    n_values = [1000, 5000, 10000, 20000, 50000, 100000, 150000]
    thread_count = 4

    kernel_times = []
    para_times = []
    seq_times = []

    for n in n_values:
        print(f"\nSzámítások n = {n}...")

        print("\nKernel:")
        _, kernel_time = opencl_sin_maclaurin(
            x, n, 4)
        kernel_times.append(kernel_time)

        print("Párhuzamosított:")
        _, para_time = sin_maclaurin_parallel(
            x, n, thread_count)
        para_times.append(para_time)

        print("\nSzekvenciális:")
        _, seq_time = seq_sin_maclaurin(x, n)
        seq_times.append(seq_time)

    plt.figure(figsize=(10, 6))
    plt.plot(n_values, kernel_times, label='Kernel', marker='o')
    plt.plot(n_values, para_times, label='Párhuzamosított', marker='s')
    plt.plot(n_values, seq_times, label='Szekvenciális', marker='^')

    plt.title('Futási idők különböző számítási módszerekkel')
    plt.xlabel('n értéke')
    plt.ylabel('Futási idő (s)')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks(n_values, labels=n_values)
    plt.tight_layout()
    plt.show()

    for i in range(len(n_values)):
        print(
            f"n = {n_values[i]}:\n\tKernel: {kernel_times[i]}\n\tPárhuzamosított: {para_times[i]}\n\tSzekvenciális: {seq_times[i]}")


if __name__ == "__main__":
    main()
