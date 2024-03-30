__kernel void maclaurin_sin(__global float* result, const float x, const int start, const int end) {
    int i = get_global_id(0) + start;
    if (i >= end) return;
    
    float term = 1.0;
    int sign = 1 - 2 * (i % 2);
    for (int j = 1; j <= 2 * i + 1; j++) {
        term *= x / j;
    }
    result[i - start] = sign * term;
}