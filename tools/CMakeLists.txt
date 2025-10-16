# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.


# Generate Query Variants Tool
add_executable(generate_query_variants generate_query_variants.cpp)
target_link_libraries(generate_query_variants ${PROJECT_NAME} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS} Boost::program_options)

