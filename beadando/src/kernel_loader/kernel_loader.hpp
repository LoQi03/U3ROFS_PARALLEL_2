#ifndef KERNEL_LOADER_HPP
#define KERNEL_LOADER_HPP

#include <memory>

std::unique_ptr<char[]> load_kernel_source(const char *const path, int *error_code);

#endif // KERNEL_LOADER_HPP