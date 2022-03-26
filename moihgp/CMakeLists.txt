cmake_minimum_required(VERSION 2.8.3)
project(moihgp)

# Set to Release in order to speed up the program significantly
set(CMAKE_BUILD_TYPE Release) #None, Debug, Release, RelWithDebInfo, MinSizeRel

message(STATUS "System: ${CMAKE_SYSTEM}")
## System dependencies are found with CMake's conventions
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake_modules)
message(STATUS "${CMAKE_MODULE_PATH}")

# Eigen3
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
set(EXTERNAL_INCLUDE_DIRS ${Eigen_INCLUDE_DIRS})

## C++11 support
IF(NOT MSVC)
  include(CheckCXXCompilerFlag)
  CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
  CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
  if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
  else()
    message(
      STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support."
    )
  endif()
endif()

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(include)
include_directories(SYSTEM)

## Build additional executables

add_executable(example_online_learning cpp_examples/example_online_learning.cpp)
target_link_libraries(example_online_learning
  ${EXTERNAL_LIBS}
  Eigen3::Eigen
  pthread
)

add_executable(example_regression cpp_examples/example_regression.cpp)
target_link_libraries(example_regression
  ${EXTERNAL_LIBS}
  Eigen3::Eigen
  pthread
)

add_library(moihgp MODULE src/wrapper.cpp)
target_link_libraries(moihgp
  ${EXTERNAL_LIBS}
  Eigen3::Eigen
  pthread
)
