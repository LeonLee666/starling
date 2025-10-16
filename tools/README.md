# DiskANN Tools

This directory contains auxiliary tools for the DiskANN/Starling project.

## Available Tools

### Generate Query Variants

A standalone tool for generating query variants with controlled noise perturbations.

**Quick Start:**
```bash
# Build
mkdir -p build && cd build
cmake ..
make generate_query_variants

# Run
./generate_query_variants --data_type float \
                         -i queries.bin \
                         -o queries_expanded.bin \
                         -n 10
```

**Features:**
- Supports `float`, `int8`, `uint8` data types
- Configurable variant count per query
- Smart noise generation (percentage-based for float, additive for integers)
- Automatic boundary checking for integer types

**Documentation:**
- [Detailed Documentation](README_generate_query_variants.md) - Complete usage guide
- [Example Script](example_usage.sh) - Shell script with examples

**Files:**
- `generate_query_variants.cpp` - Main C++ source code
- `README_generate_query_variants.md` - Detailed documentation
- `example_usage.sh` - Usage examples
- `CMakeLists.txt` - Build configuration

## Building

The tools are automatically built when you build the project:

```bash
cd /path/to/starling
mkdir -p build && cd build
cmake ..
make
```

To build only the tools:

```bash
make generate_query_variants
```

## Usage

### Generate Query Variants

Generate perturbed variants of input queries for testing index robustness:

```bash
# Basic usage
./build/generate_query_variants \
    --data_type float \
    --input_query_file input.bin \
    --output_query_file output.bin \
    --variant_num 10

# Short form
./build/generate_query_variants --data_type float -i input.bin -o output.bin -n 10

# Custom noise parameters
./build/generate_query_variants --data_type float -i input.bin -o output.bin -n 10 \
    --noise_ratio_float 0.1

./build/generate_query_variants --data_type uint8 -i input.bin -o output.bin -n 5 \
    --noise_range_int 3
```

**Parameters:**
- `--data_type`: Data type (`float`, `int8`, `uint8`)
- `-i, --input_query_file`: Input query file (DiskANN binary format)
- `-o, --output_query_file`: Output query file (DiskANN binary format)
- `-n, --variant_num`: Number of variants per query
- `--noise_ratio_float`: Noise ratio for float (default: 0.05)
- `--noise_range_int`: Noise range for integers (default: 2)

**Example Workflow:**
```bash
# 1. Generate expanded queries
./build/generate_query_variants --data_type float \
    -i sift_query.bin -o sift_query_expanded.bin -n 10

# 2. Use with search_disk_index
./build/search_disk_index \
    --data_type float \
    --dist_fn l2 \
    --index_path_prefix sift_index \
    --query_file sift_query_expanded.bin \
    --result_path results \
    --recall_at 10 \
    --search_list 50 100 \
    --disk_file_path sift_disk.index
```

## Use Cases

### Query Variants
- **Testing index robustness**: Evaluate performance with noisy queries
- **Generating test datasets**: Expand small query sets for comprehensive testing
- **Simulating real-world scenarios**: Model query variations in production
- **Performance analysis**: Study the impact of noise on recall and latency

## Contributing

When adding new tools:
1. Place source files in the `tools/` directory
2. Add build target to `tools/CMakeLists.txt`
3. Create documentation (README or inline help)
4. Provide usage examples

## License

Copyright (c) Microsoft Corporation. All rights reserved.  
Licensed under the MIT license.

