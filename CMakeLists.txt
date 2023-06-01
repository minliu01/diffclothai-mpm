cmake_minimum_required(VERSION 3.15)
project(DiffCloth)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "set env var")

set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_STANDARD 17)
set(EXECUTABLE_NAME DiffCloth)
set(EXTERNAL_ROOT ${PROJECT_SOURCE_DIR}/external)
set(LIB dependencies/lib)

#nanogui
set(NANOGUI_EIGEN_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/external/eigen")
set(NANOGUI_BUILD_EXAMPLE OFF CACHE BOOL " " FORCE)
set(NANOGUI_BUILD_PYTHON OFF CACHE BOOL " " FORCE)
set(NANOGUI_INSTALL OFF CACHE BOOL " " FORCE)

add_definitions(-DSOURCE_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}\")

if (UNIX)
    set(_GLFW_X11 1)
    add_definitions(-D_GLFW_X11=1)
    message(STATUS "Using X11 for window creation")
else()
    message(FATAL_ERROR "No supported platform was detected")
endif()

add_definitions(-DCMAKE_TOOLCHAIN_FILE=${PROJECT_SOURCE_DIR}/external/vcpkg/scripts/buildsystems/vcpkg.cmake)

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

if (USE_DEBUG)
    add_compile_options(-DUSE_DEBUG)
endif ()

message("cxx Flags: " ${CMAKE_CXX_FLAGS})

source_group("Shaders" FILES src/gpu/shaders)
source_group("Assets" FILES src/assets)

add_subdirectory(external/nanogui)
add_subdirectory(${EXTERNAL_ROOT})

set_property(TARGET glfw glfw_objects nanogui PROPERTY FOLDER "dependencies")
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
add_definitions(${NANOGUI_EXTRA_DEFS})

###############
get_directory_property(EXTERNAL_HEADER
        DIRECTORY ${EXTERNAL_ROOT}
        DEFINITION EXTERNAL_HEADER)

include_directories(${NANOGUI_EXTRA_INCS})
include_directories(dependencies/include)
include_directories(src/assets)
include_directories(src/gpu)
include_directories(external/nanogui/include)
include_directories(external/LBFGSpp/include)
include_directories(/usr/local/include)

include_directories(dependencies/lib)
include_directories(external/Tight-Inclusion/src/)
include_directories(external/Tight-Inclusion/src/tight_inclusion/)
include_directories(external/Tight-Inclusion/src/tight_inclusion/rational/)

link_directories(${LIB})

# OmegaEngine cpp
file(GLOB HEADERS "src/code/*.h" "src/code/engine/*.h" "src/code/simulation/*.h" "src/code/supports/*.h" "src/code/optimization/*.h")
file(GLOB SRCFILES "src/code/main.cpp" "src/code/*.h" "src/code/*.c" "src/code/engine/*.cpp" "src/code/engine/*.h" "src/code/simulation/*.cpp" "src/code/simulation/*.h" "src/code/supports/*.cpp" "src/code/supports/*.h" "src/code/optimization/*.cpp" "src/code/optimization/*.h")
source_group("Sources" FILES ${SRCFILES})
add_executable(${EXECUTABLE_NAME} ${SRCFILES}  )

target_link_libraries(${EXECUTABLE_NAME} nanogui ${NANOGUI_EXTRA_LIBS})
target_link_libraries(${EXECUTABLE_NAME} OpenMP::OpenMP_CXX)
target_link_libraries(${EXECUTABLE_NAME} fcl)
target_link_libraries(${EXECUTABLE_NAME} ccd)
target_link_libraries(${EXECUTABLE_NAME} openvdb)
target_link_libraries(${EXECUTABLE_NAME} tbb)
target_include_directories(${EXECUTABLE_NAME} PRIVATE ${EXTERNAL_HEADER})

#find_package(PythonLibs 3.8 REQUIRED )
find_package(Python 3.8 COMPONENTS Interpreter Development)
#message(${PYTHON_INCLUDE_DIRS})
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(external/pybind11/include)
add_subdirectory(external/pybind11)


pybind11_add_module(diffcloth_py "src/code/python_interface.cpp" ${HEADERS} ${SRCFILES})
# target_link_libraries(diffcloth_py PRIVATE ${PYTHON_LIBRARIES})
target_link_libraries(diffcloth_py PRIVATE pybind11::module)
target_link_libraries(diffcloth_py PRIVATE nanogui ${NANOGUI_EXTRA_LIBS})
target_link_libraries(diffcloth_py PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(diffcloth_py PRIVATE fcl)
target_link_libraries(diffcloth_py PRIVATE ccd)
target_link_libraries(diffcloth_py PRIVATE openvdb)
target_link_libraries(diffcloth_py PRIVATE tbb)
target_include_directories(diffcloth_py PRIVATE ${EXTERNAL_HEADER})
