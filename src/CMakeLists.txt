# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

set(CMAKE_CXX_STANDARD 17)
find_package(folly REQUIRED)
if(MSVC)
    add_subdirectory(dll)
else()
    #file(GLOB CPP_SOURCES *.cpp)
    set(CPP_SOURCES ann_exception.cpp aux_utils.cpp distance.cpp index.cpp
        linux_aligned_file_reader.cpp math_utils.cpp natural_number_map.cpp
        natural_number_set.cpp memory_mapper.cpp partition_and_pq.cpp
        pq_flash_index.cpp logger.cpp utils.cpp page_search.cpp visit_freq.cpp
        range_search.cpp io_merge_wrapper.cpp lightweight_io_merger.cpp)
    list(APPEND CPP_SOURCES page_pool.cpp)
    
    # Create SHARED library (dynamic) to properly handle folly dependencies
    add_library(${PROJECT_NAME} SHARED ${CPP_SOURCES})
    add_library(${PROJECT_NAME}_s STATIC ${CPP_SOURCES})
    
    # Link libraries - folly dependencies will be resolved in the shared library
    target_link_libraries(${PROJECT_NAME} PUBLIC onetbb::onetbb aio folly::folly)
    target_link_libraries(${PROJECT_NAME}_s PUBLIC onetbb::onetbb aio folly::folly)
    
    # Ensure folly symbols are properly exported from shared library
    target_link_options(${PROJECT_NAME} PRIVATE -Wl,--no-as-needed)
endif()
install()
