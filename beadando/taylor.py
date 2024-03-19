import math
import time
from multiprocessing import Pool
import pyopencl as cl
import numpy as np

kernel_code = """
__kernel void taylor_cos(__global float* result, float x, int n) {
    int i = get_global_id(0);
    float term = 1.0;
    for (int j = 1; j <= 2 * i; j++) {
        term *= j;
    }
    term = pow(-1.0f, i) * pow(x, 2 * i) / term;
    if (i < n) {
        result[i] = term;
    }
}

__kernel void taylor_sin(__global float* result, float x, int n) {
    int i = get_global_id(0);
    float term = 1.0;
    for (int j = 1; j <= 2 * i + 1; j++) {
        term *= j;
    }
    term = pow(-1.0f, i) * pow(x, 2 * i + 1) / term;
    if (i < n) {
        result[i] = term;
    }
}

__kernel void taylor_exp(__global float* result, float x, int n) {
    int i = get_global_id(0);
    float term = 1.0;
    for (int j = 1; j <= i; j++) {
        term *= j;
    }
    term = pow(x, i) / term;
    if (i < n) {
        result[i] = term;
    }
}
"""

def run_opencl_kernel(kernel_code, kernel_name, x, n):
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    context = cl.Context([device])
    queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)
    
    program = cl.Program(context, kernel_code).build()

    result = np.zeros(n, dtype=np.float32)

    result_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, result.nbytes)

    kernel = getattr(program, kernel_name)
    
    event = kernel(queue, result.shape, None, result_buffer, np.float32(x), np.int32(n))
    
    cl.enqueue_copy(queue, result, result_buffer).wait()

    start_time = event.get_profiling_info(cl.profiling_info.START)
    end_time = event.get_profiling_info(cl.profiling_info.END)
    execution_time = end_time - start_time

    return result, execution_time

def test_opencl_taylor(func, x, n):
    result,execution_time = run_opencl_kernel(kernel_code, f"taylor_{func}", x, n)
    print(f"taylor_{func}({x}) = {np.sum(result)}\t ido: {execution_time/1e6}")

def factorial(n):
    result = 1.0
    for i in range(2, n + 1):
        result *= i
    return result

def cos_taylor(x, n, thread_count):
    start = time.time()
    result = 1.0
    with Pool(processes=thread_count) as pool:
        terms = pool.starmap(compute_cos_term, [(x, i) for i in range(1, n + 1)])
        for term in terms:
            result += term
    print(f"cos({x}) = {result}\t ido: {time.time() - start}")
    return result

def compute_cos_term(x, i):
    return math.pow(-1, i) * math.pow(x, 2 * i) / factorial(2 * i)

def sin_taylor(x, n, thread_count):
    start = time.time()
    result = x
    with Pool(processes=thread_count) as pool:
        terms = pool.starmap(compute_sin_term, [(x, i) for i in range(1, n + 1)])
        for term in terms:
            result += term
    print(f"sin({x}) = {result}\t ido: {time.time() - start}")
    return result

def compute_sin_term(x, i):
    return math.pow(-1, i) * math.pow(x, 2 * i + 1) / factorial(2 * i + 1)

def exp_taylor(x, n, thread_count):
    start = time.time()
    result = 1.0
    with Pool(processes=thread_count) as pool:
        terms = pool.starmap(compute_exp_term, [(x, i) for i in range(1, n + 1)])
        for term in terms:
            result += term
    print(f"exp({x}) = {result}\t ido: {time.time() - start}")
    return result

def compute_exp_term(x, i):
    return math.pow(x, i) / factorial(i)

def seq_cos_taylor(x, n):
    start = time.time()
    result = 1.0
    for i in range(1, n + 1):
        term = math.pow(-1, i) * math.pow(x, 2 * i) / factorial(2 * i)
        result += term
    print(f"cos({x}) = {result}\t ido: {time.time() - start}")
    return result

def seq_sin_taylor(x, n):
    start = time.time()
    result = x
    for i in range(1, n + 1):
        term = math.pow(-1, i) * math.pow(x, 2 * i + 1) / factorial(2 * i + 1)
        result += term
    print(f"sin({x}) = {result}\t ido: {time.time() - start}")
    return result

def seq_exp_taylor(x, n):
    start = time.time()
    result = 1.0
    for i in range(1, n + 1):
        term = math.pow(x, i) / factorial(i)
        result += term
    print(f"exp({x}) = {result}\t ido: {time.time() - start}")
    return result

def main():
    x = 1.0
    n = 10000
    thread_count = 7

    print("Párhuzamosított Taylor sorok:")
    cos_taylor(x, n, thread_count)
    sin_taylor(x, n, thread_count)
    exp_taylor(x, n, thread_count)

    print("\nSzekvenciális Taylor sorok:")
    seq_cos_taylor(x, n)
    seq_sin_taylor(x, n)
    seq_exp_taylor(x, n)

    print("\nKernel Taylor sorok:")
    test_opencl_taylor("cos", x, n)
    test_opencl_taylor("sin", x, n)
    test_opencl_taylor("exp", x, n)

if __name__ == "__main__":
    main()

