#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <fstream>
#include <sstream>
#include <chrono>
#include "src/kernel_loader/kernel_loader.hpp"
#include <CL/cl.h>
#define MAX_SOURCE_SIZE (0x100000)
using namespace std;

vector<pair<double, double>> read_tsp_file(const string &filename)
{
    vector<pair<double, double>> tsp;
    ifstream file(filename);
    string line;
    bool start_reading = false;

    while (getline(file, line))
    {
        if (line == "NODE_COORD_SECTION")
        {
            start_reading = true;
            continue;
        }
        if (start_reading && line != "EOF")
        {
            stringstream ss(line);
            int index;
            double x, y;
            ss >> index >> x >> y;
            tsp.push_back({x, y});
        }
    }
    file.close();
    return tsp;
}

double distance(pair<double, double> c1, pair<double, double> c2)
{
    return sqrt(pow(c2.first - c1.first, 2) + pow(c2.second - c1.second, 2));
}

double calc_fitness(const vector<pair<double, double>> &tsp, const vector<int> &order)
{
    double dist = 0;
    for (size_t i = 0; i < order.size() - 1; ++i)
    {
        dist += distance(tsp[order[i]], tsp[order[i + 1]]);
    }
    dist += distance(tsp[order.back()], tsp[order.front()]);
    return dist;
}

vector<int> two_opt_swap(const vector<int> &order, int i, int j)
{
    vector<int> new_order = order;
    int size = new_order.size();

    int count = (j - i + 1) / 2;
    for (int k = 0; k < count; ++k)
    {
        swap(new_order[(i + k) % size], new_order[(j - k) % size]);
    }
    return new_order;
}

vector<int> local_search(const vector<pair<double, double>> &tsp, const vector<int> &order, int iterations)
{
    vector<int> best_order = order;
    double best_dist = calc_fitness(tsp, order);
    vector<int> new_order;

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dis(0, order.size() - 1);

    for (int iter = 0; iter < iterations; ++iter)
    {
        int i = dis(gen);
        int j = dis(gen);
        if (i == j)
            continue;
        if (i > j)
            swap(i, j);

        new_order = two_opt_swap(best_order, i, j);
        double new_dist = calc_fitness(tsp, new_order);

        if (new_dist < best_dist)
        {
            best_dist = new_dist;
            best_order = new_order;
        }
    }
    return best_order;
}

void run_tsp_no_kernel(vector<pair<double, double>> &tsp, vector<int> &initial_order, int iterations = 1000)
{
    auto start_time = chrono::high_resolution_clock::now();
    vector<int> best_order = local_search(tsp, initial_order, iterations);

    cout << "Best order: ";
    for (int city : best_order)
    {
        cout << city << " ";
    }
    cout << endl;
    cout << "Total distance: " << calc_fitness(tsp, best_order) << endl;

    auto end_time = chrono::high_resolution_clock::now();

    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
    cout << "Execution time: " << duration << " milliseconds" << endl;
}

void run_tsp_kernel(std::vector<std::pair<double, double>> &tsp, std::vector<int> &initial_order, int iterations = 1000)
{
    // OpenCL setup
    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);

    // Create an OpenCL context
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

    // Create memory buffers on the device for each vector
    cl_mem tsp_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, tsp.size() * sizeof(std::pair<double, double>), NULL, &ret);
    cl_mem order_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, initial_order.size() * sizeof(int), NULL, &ret);
    cl_mem new_order_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, initial_order.size() * sizeof(int), NULL, &ret);
    cl_mem best_dist_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(double), NULL, &ret);

    // Copy the lists tsp and initial_order to their respective memory buffers
    ret = clEnqueueWriteBuffer(command_queue, tsp_mem_obj, CL_TRUE, 0, tsp.size() * sizeof(std::pair<double, double>), &tsp[0], 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, order_mem_obj, CL_TRUE, 0, initial_order.size() * sizeof(int), &initial_order[0], 0, NULL, NULL);

    // Load the kernel source code into the array source_str
    const char filename[] = "src/kernels/local_search.cl"; // replace with your kernel file name
    size_t source_size;
    char *source_str;
    FILE *fp = fopen(filename, "r");
    if (!fp)
    {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char *)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);

    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);

    // Build the program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "local_search", &ret);
    int tsp_size = tsp.size();
    int initial_order_size = initial_order.size();
    // Set the arguments of the kernel
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&tsp_mem_obj);
    ret = clSetKernelArg(kernel, 1, sizeof(int), &tsp_size);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&order_mem_obj);
    ret = clSetKernelArg(kernel, 3, sizeof(int), &initial_order_size);
    ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&new_order_mem_obj);
    ret = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&best_dist_mem_obj);

    // Execute the OpenCL kernel on the list
    size_t global_item_size = initial_order.size(); // Process the entire lists
    size_t local_item_size = 64;                    // Divide work items into groups of 64
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);

    // Read the memory buffer order_mem_obj on the device to the local variable order
    int *order = new int[initial_order.size()];
    double best_dist;
    ret = clEnqueueReadBuffer(command_queue, order_mem_obj, CL_TRUE, 0, initial_order.size() * sizeof(int), order, 0, NULL, NULL);
    ret = clEnqueueReadBuffer(command_queue, best_dist_mem_obj, CL_TRUE, 0, sizeof(double), &best_dist, 0, NULL, NULL);
    // Display the result to the screen
    for (int i = 0; i < initial_order.size(); i++)
        std::cout << order[i] << " ";
    std::cout << std::endl;
    std::cout << "Best distance: " << best_dist << std::endl;

    // Clean up
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(tsp_mem_obj);
    ret = clReleaseMemObject(order_mem_obj);
    ret = clReleaseMemObject(new_order_mem_obj);
    ret = clReleaseMemObject(best_dist_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    delete[] order;
}

int main()
{
    cout << "Reading tsp file..." << endl;
    vector<pair<double, double>> tsp = read_tsp_file("src/tsp_problems/xqg237.tsp");

    vector<int> initial_order(tsp.size());
    iota(initial_order.begin(), initial_order.end(), 0);
    printf("Running TSP(kernel)\n");
    run_tsp_kernel(tsp, initial_order);
    printf("\nRunning TSP(no kernel)\n");
    run_tsp_no_kernel(tsp, initial_order);
    return 0;
}
