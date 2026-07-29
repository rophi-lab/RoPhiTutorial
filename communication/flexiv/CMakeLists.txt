cmake_minimum_required(VERSION 3.16)

project(flexiv_bridge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(flexiv_rdk REQUIRED)

# Locate LCM by searching for its headers and library directly rather than via
# pkg-config. The active conda env ships a pip-installed lcm whose headers and
# library live under the Python site-packages tree and ship no pkg-config file;
# pkg-config therefore falls back to the system liblcm-dev under /usr/include,
# which is a different build that the conda toolchain cannot consume (and whose
# include dir CMake/pkg-config strip from the flags anyway). Search the active
# env explicitly so we pick the copy that matches the compiler.
find_path(LCM_INCLUDE_DIR
    NAMES lcm/lcm.h
    HINTS $ENV{CONDA_PREFIX}
    PATH_SUFFIXES
        include
        lib/python3.10/site-packages/include
        lib/python3.11/site-packages/include
        lib/python3.12/site-packages/include
)
find_library(LCM_LIBRARY
    NAMES lcm
    HINTS $ENV{CONDA_PREFIX}
    PATH_SUFFIXES
        lib
        lib/python3.10/site-packages/lib
        lib/python3.11/site-packages/lib
        lib/python3.12/site-packages/lib
)
if(NOT LCM_INCLUDE_DIR OR NOT LCM_LIBRARY)
    message(FATAL_ERROR
        "LCM not found (LCM_INCLUDE_DIR=${LCM_INCLUDE_DIR}, "
        "LCM_LIBRARY=${LCM_LIBRARY}). Activate the conda env that provides lcm "
        "(e.g. `conda activate manipulator`, with `pip install lcm`) before "
        "configuring.")
endif()
message(STATUS "Found LCM: ${LCM_LIBRARY} (include dir: ${LCM_INCLUDE_DIR})")

add_library(
    flexiv_bridge_core
    src/BridgeConfig.cpp
    src/FlexivBridge.cpp
    src/FlexivClient.cpp
    src/JointMapper.cpp
    src/LcmCommandSubscriber.cpp
    src/LcmStatePublisher.cpp
    src/SafetyMonitor.cpp
)

target_include_directories(
    flexiv_bridge_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${LCM_INCLUDE_DIR}
)

target_link_libraries(
    flexiv_bridge_core
    PUBLIC
        yaml-cpp
        Threads::Threads
        flexiv::flexiv_rdk
        ${LCM_LIBRARY}
)

add_executable(flexiv_bridge src/main.cpp)
target_link_libraries(flexiv_bridge PRIVATE flexiv_bridge_core)

add_executable(rt_smoke_test examples/rt_smoke_test.cpp)
target_link_libraries(rt_smoke_test PRIVATE flexiv_bridge_core)

# Dumps Flexiv's identified gravity model across a joint sweep for comparison
# against the URDF (see scripts/compare_rdk_urdf.py).
add_executable(rdk_dynamics_dump tools/rdk_dynamics_dump.cpp)
target_link_libraries(rdk_dynamics_dump PRIVATE flexiv_bridge_core)
