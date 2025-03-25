cmake_minimum_required(VERSION 3.10)

# Project name and C++ standard
project(ShomerLibrary VERSION 1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_INSTALL_LIBDIR "/usr/lib")
set(CMAKE_INSTALL_INCLUDEDIR "/usr/include")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fsee -fomit-frame-pointer -fno-signed-zeros -fno-math-errno -funroll-loops")

# Include directories (Ensure headers are found)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/ShomerLib)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(Threads REQUIRED)
# Collect all source files
file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

# Create a static library
add_library(ShomerLib STATIC ${SOURCES})

# Properly set include directories for this library
target_include_directories(ShomerLib PUBLIC 
    ${CMAKE_SOURCE_DIR}/include/ShomerLib # Ensure include directory is set
    ${OpenCV_INCLUDE_DIRS} 
    ${EIGEN3_INCLUDE_DIRS} 
    ${CERES_INCLUDE_DIRS}
)

# Ensure correct linking
if(TARGET Ceres::ceres)
    target_link_libraries(ShomerLib 
        ${OpenCV_LIBS} 
        Eigen3::Eigen 
        Ceres::ceres
        ${CMAKE_THREAD_LIBS_INIT}
    )
else()
    message(WARNING "Ceres::ceres target not found. Linking manually.")
    target_link_libraries(ShomerLib 
        ${OpenCV_LIBS} 
        Eigen3::Eigen 
        ${CERES_LIBRARIES}
        ${CMAKE_THREAD_LIBS_INIT}
    )
endif()

# Installation rules (optional)
install(TARGETS ShomerLib DESTINATION lib)
if(EXISTS "${CMAKE_SOURCE_DIR}/include")
    install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/ DESTINATION include)
endif()


