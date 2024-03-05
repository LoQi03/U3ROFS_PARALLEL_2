__kernel void vector_addition(__global const float4* a, __global const float4* b, __global float4* result, int n)
{
    int i = get_global_id(0);
    if (i < n) {
        result[i] = a[i] + b[i];
    }
}
