__kernel void local_search(__global double2 *tsp, int tsp_size, __global int *order, int order_size, __global int *new_order, __global double *best_dist)
{
    // Get global id
    int gid = get_global_id(0);

    // Create local copy of order
    __global int local_order[order_size];
    for (int i = 0; i < order_size; ++i) {
        local_order[i] = order[i];
    }

    // Get random indexes
    int i = gid % order_size;
    int j = (gid / order_size) % order_size;

    // Swap indexes if necessary
    if (i > j) {
        int temp = i;
        i = j;
        j = temp;
    }

    // Perform 2-opt swap
    int count = (j - i + 1) / 2;
    for (int k = 0; k < count; ++k) {
        int temp = local_order[(i + k) % order_size];
        local_order[(i + k) % order_size] = local_order[(j - k) % order_size];
        local_order[(j - k) % order_size] = temp;
    }

    // Calculate distance
    double dist = 0.0;
    for (int idx = 0; idx < order_size - 1; ++idx) {
        int city1 = local_order[idx];
        int city2 = local_order[idx + 1];
        dist += sqrt(pow(tsp[city2].x - tsp[city1].x, 2) + pow(tsp[city2].y - tsp[city1].y, 2));
    }
    dist += sqrt(pow(tsp[local_order[order_size - 1]].x - tsp[local_order[0]].x, 2) + pow(tsp[local_order[order_size - 1]].y - tsp[local_order[0]].y, 2));

    // Update best order and distance if necessary
    if (dist < *best_dist) {
        *best_dist = dist;
        for (int idx = 0; idx < order_size; ++idx) {
            new_order[idx] = local_order[idx];
        }
    }
}
