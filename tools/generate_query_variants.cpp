// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include <string>
#include <random>
#include <cstring>
#include <boost/program_options.hpp>

#include "utils.h"

namespace po = boost::program_options;

template<typename T>
int generate_query_variants(
    const std::string& input_query_file,
    const std::string& output_query_file,
    size_t variant_num,
    float noise_ratio_float = 0.05f,
    int noise_range_int = 2) {
  
  std::cout << "Loading input query file: " << input_query_file << std::endl;
  
  // Load original queries
  T* original_query = nullptr;
  size_t query_num, query_dim, query_aligned_dim;
  diskann::load_aligned_bin<T>(input_query_file, original_query, query_num, 
                               query_dim, query_aligned_dim);
  
  std::cout << "Loaded " << query_num << " queries with dimension " 
            << query_dim << " (aligned: " << query_aligned_dim << ")" << std::endl;
  
  // Calculate expanded query count
  size_t original_query_num = query_num;
  size_t expanded_query_num = query_num * (variant_num + 1);
  
  std::cout << "Generating " << variant_num << " variants per query..." << std::endl;
  
  // Allocate aligned memory for expanded queries
  T* expanded_query = nullptr;
  diskann::alloc_aligned(((void**) &expanded_query),
                        expanded_query_num * query_aligned_dim * sizeof(T),
                        8 * sizeof(T));
  
  // Random number generator for noise generation
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> dim_dist(0, query_dim - 1);
  
  // Generate expanded queries with variants
  for (size_t i = 0; i < original_query_num; i++) {
    // Copy original query to position i*(variant_num + 1)
    std::memcpy(expanded_query + (i * (variant_num + 1)) * query_aligned_dim,
                original_query + i * query_aligned_dim,
                query_aligned_dim * sizeof(T));
    
    // Generate variants
    for (size_t v = 1; v <= variant_num; v++) {
      size_t target_idx = i * (variant_num + 1) + v;
      
      // Copy original query as base
      std::memcpy(expanded_query + target_idx * query_aligned_dim,
                  original_query + i * query_aligned_dim,
                  query_aligned_dim * sizeof(T));
      
      // Randomly select one dimension to perturb
      size_t perturb_dim = dim_dist(gen);
      T original_value = original_query[i * query_aligned_dim + perturb_dim];
      
      // Apply type-appropriate noise to the selected dimension
      if (std::is_same<T, float>::value) {
        // For float type: apply percentage-based noise
        std::uniform_real_distribution<float> noise_dist(-noise_ratio_float, noise_ratio_float);
        float noise = noise_dist(gen);
        expanded_query[target_idx * query_aligned_dim + perturb_dim] = 
            static_cast<T>(static_cast<float>(original_value) * (1.0f + noise));
      } else {
        // For integer types: apply small additive noise
        std::uniform_int_distribution<int> int_noise_dist(-noise_range_int, noise_range_int);
        int noise = int_noise_dist(gen);
        
        // Ensure the value stays within valid range for the type
        int new_value = static_cast<int>(original_value) + noise;
        if (std::is_same<T, uint8_t>::value) {
          new_value = std::max(0, std::min(255, new_value));
        } else if (std::is_same<T, int8_t>::value) {
          new_value = std::max(-128, std::min(127, new_value));
        }
        expanded_query[target_idx * query_aligned_dim + perturb_dim] = static_cast<T>(new_value);
      }
    }
    
    // Progress indicator
    if ((i + 1) % 1000 == 0 || i == original_query_num - 1) {
      std::cout << "Processed: " << (i + 1) << "/" << original_query_num << " queries\r" << std::flush;
    }
  }
  std::cout << std::endl;
  
  // Save expanded queries
  std::cout << "Saving expanded queries to: " << output_query_file << std::endl;
  diskann::save_bin<T>(output_query_file, expanded_query, expanded_query_num, query_dim);
  
  // Free memory
  diskann::aligned_free(expanded_query);
  diskann::aligned_free(original_query);
  
  std::cout << "Query expansion complete: " << original_query_num 
            << " -> " << expanded_query_num << " queries" << std::endl;
  std::cout << "Output saved to: " << output_query_file << std::endl;
  
  return 0;
}

int main(int argc, char** argv) {
  std::string data_type, input_query_file, output_query_file;
  size_t variant_num;
  float noise_ratio_float;
  int noise_range_int;
  
  po::options_description desc{"Generate Query Variants Tool\n\n"
                                "This tool generates variants of input queries by adding small perturbations.\n"
                                "Each original query will be expanded to (variant_num + 1) queries.\n\n"
                                "Arguments"};
  
  try {
    desc.add_options()
      ("help,h", "Print this help message")
      ("data_type", 
       po::value<std::string>(&data_type)->required(),
       "Data type of queries: int8, uint8, or float")
      ("input_query_file,i",
       po::value<std::string>(&input_query_file)->required(),
       "Path to input query file (binary format)")
      ("output_query_file,o",
       po::value<std::string>(&output_query_file)->required(),
       "Path to output query file (binary format)")
      ("variant_num,n",
       po::value<size_t>(&variant_num)->required(),
       "Number of variants to generate per query")
      ("noise_ratio_float",
       po::value<float>(&noise_ratio_float)->default_value(0.05f),
       "Noise ratio for float type (default: 0.05 = 5%)")
      ("noise_range_int",
       po::value<int>(&noise_range_int)->default_value(2),
       "Noise range for integer types, ±noise_range_int (default: 2)");
    
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    
    if (vm.count("help")) {
      std::cout << desc << std::endl;
      std::cout << "\nExample usage:\n";
      std::cout << "  ./generate_query_variants --data_type float --input_query_file queries.bin \\\n";
      std::cout << "                            --output_query_file queries_expanded.bin --variant_num 10\n";
      std::cout << "\n";
      std::cout << "  ./generate_query_variants --data_type uint8 -i queries.bin \\\n";
      std::cout << "                            -o queries_expanded.bin -n 5 --noise_range_int 3\n";
      return 0;
    }
    
    po::notify(vm);
  } catch (const std::exception& ex) {
    std::cerr << "Error: " << ex.what() << "\n\n";
    std::cerr << desc << std::endl;
    return -1;
  }
  
  // Validate parameters
  if (variant_num == 0) {
    std::cerr << "Error: variant_num must be greater than 0" << std::endl;
    return -1;
  }
  
  if (noise_ratio_float < 0 || noise_ratio_float > 1.0f) {
    std::cerr << "Error: noise_ratio_float should be in range [0, 1]" << std::endl;
    return -1;
  }
  
  if (noise_range_int < 0) {
    std::cerr << "Error: noise_range_int should be non-negative" << std::endl;
    return -1;
  }
  
  // Process based on data type
  try {
    if (data_type == "float") {
      return generate_query_variants<float>(
          input_query_file, output_query_file, variant_num, 
          noise_ratio_float, noise_range_int);
    } else if (data_type == "int8") {
      return generate_query_variants<int8_t>(
          input_query_file, output_query_file, variant_num, 
          noise_ratio_float, noise_range_int);
    } else if (data_type == "uint8") {
      return generate_query_variants<uint8_t>(
          input_query_file, output_query_file, variant_num, 
          noise_ratio_float, noise_range_int);
    } else {
      std::cerr << "Error: Unsupported data type '" << data_type << "'. "
                << "Supported types: float, int8, uint8" << std::endl;
      return -1;
    }
  } catch (const std::exception& e) {
    std::cerr << "Error during query variant generation: " << e.what() << std::endl;
    return -1;
  }
  
  return 0;
}

