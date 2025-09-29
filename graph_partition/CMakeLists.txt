cmake_minimum_required(VERSION 3.16)
project(SSD_Based_Plan)
set(CMAKE_CXX_STANDARD 17)
include_directories(include)


# OpenMP
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(FATAL_ERROR "no OpenMP supprot")
endif()

#boost

find_package(Boost COMPONENTS program_options REQUIRED)
find_package(folly REQUIRED)

# Force linking of all libraries to resolve folly symbols
if(NOT MSVC)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--no-as-needed")
endif()

add_executable(partitioner src/partitioner.cpp)
target_link_libraries(partitioner PUBLIC Boost::program_options diskann onetbb::onetbb folly::folly)

# Enable AVX2 for this target on GCC/Clang to avoid ABI warnings and ensure
# availability of 256-bit intrinsics used in headers.
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(partitioner PRIVATE -mavx2 -mfma)
    target_compile_definitions(partitioner PRIVATE USE_AVX2)
endif()

