cmake_minimum_required(VERSION 2.8.12.2)

add_definitions(-DWITH_AVX2)
add_compile_options(-mavx)

project(matrix)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11") # set C++ 11
# set(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -std=c99")
elseif(COMPILER_SUPPORTS_CXX0X)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
message(STATUS "The comipler ${CMAKE_CXX_COMIPLER} has no C++ 11 suport. Please use a different C++ comipler.")
endif()

aux_source_directory(. DIR_SRCS)

add_executable(main ${DIR_SRCS})
