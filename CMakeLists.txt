cmake_minimum_required(VERSION 3.19)
project(dicecalc)

set(CMAKE_CXX_STANDARD 17)

# boost 1.75.0 on windows does not build the dll version of things.
if (WIN32)
    set(Boost_USE_STATIC_LIBS 1)
endif()

find_package(Boost COMPONENTS json REQUIRED)

include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIRS})

message(STATUS ${Boost_LIBRARY_DIRS})

add_executable(dicecalc main.cpp)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(dicecalc PUBLIC OpenMP::OpenMP_CXX)
    target_compile_options(dicecalc PUBLIC -openmp:llvm)
endif()