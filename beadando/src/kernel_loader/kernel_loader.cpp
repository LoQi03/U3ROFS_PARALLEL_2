#include "kernel_loader.hpp"
#include <fstream>
#include <memory>

std::unique_ptr<char[]> load_kernel_source(const char *const path, int *error_code)
{
    printf("Loading source code from file: %s\n", path);
    std::ifstream source_file(path, std::ios::binary);
    if (!source_file.is_open())
    {
        *error_code = -1;
        return nullptr;
    }

    source_file.seekg(0, std::ios::end);
    std::streampos file_size = source_file.tellg();
    source_file.seekg(0, std::ios::beg);

    std::unique_ptr<char[]> source_code(new char[file_size + 1]);

    source_file.read(source_code.get(), file_size);
    source_code[file_size] = '\0';

    *error_code = 0;
    return source_code;
}
