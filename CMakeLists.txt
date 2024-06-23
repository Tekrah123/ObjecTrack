cmake_minimum_required(VERSION 3.5)
project(pixel_tenshi)

# Set the C++ standard to C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # Ensure -std=c++11 not -std=gnu++11

# Find the OpenCV package
find_package(OpenCV REQUIRED)

# Add include directories for OpenCV
include_directories(${OpenCV_INCLUDE_DIRS})

# Define the executable for the project
add_executable(${PROJECT_NAME} src/main.cpp)

# Link OpenCV libraries with the project
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

# Add threading support explicitly for MinGW
if(MINGW)
    target_compile_options(${PROJECT_NAME} PRIVATE -pthread)
    target_link_libraries(${PROJECT_NAME} -pthread)
endif()
