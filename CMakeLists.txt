cmake_minimum_required(VERSION 3.16)
project(CostForge VERSION 0.3.2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Core library (no LLVM dependency) ────────────────

add_library(costforge_core STATIC
    src/cost_model/cost_model.cpp
    src/decision_engine/decision_engine.cpp
    src/hwprobe/hwprobe.cpp
    src/pattern_recognizer/pattern_recognizer.cpp
    src/calibrator/calibrator.cpp
    src/decision_log/decision_log.cpp
    src/interprocedural/interprocedural.cpp
    src/loop_tiling/loop_tiling.cpp
    src/auto_tuner/auto_tuner.cpp
)

target_include_directories(costforge_core PUBLIC include)
target_compile_options(costforge_core PRIVATE -Wall -Wextra -O2 -march=native -mavx2 -mfma)
# Must be position-independent: this static lib also gets linked into the
# CostForge.so LLVM pass module (COSTFORGE_LLVM_PASS=ON below).
set_target_properties(costforge_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ── Hardware probe tool ──────────────────────────────

add_executable(costforge-probe src/tools/probe_main.cpp)
target_link_libraries(costforge-probe costforge_core)

# ── Tests ────────────────────────────────────────────

enable_testing()

add_executable(test_cost_model tests/test_cost_model.cpp)
target_link_libraries(test_cost_model costforge_core)
add_test(NAME test_cost_model COMMAND test_cost_model)

add_executable(test_v020 tests/test_v020.cpp)
target_link_libraries(test_v020 costforge_core)
add_test(NAME test_v020 COMMAND test_v020)

# ── LLVM Pass (optional) ────────────────────────────

option(COSTFORGE_LLVM_PASS "Build the LLVM optimization pass" OFF)

if(COSTFORGE_LLVM_PASS)
    find_package(LLVM REQUIRED CONFIG)
    message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
    
    add_library(CostForge MODULE
        src/llvm_pass/pass.cpp
        src/transform_engine/transform_engine.cpp
        src/profile_reader/profile_reader.cpp
    )
    target_include_directories(CostForge PRIVATE ${LLVM_INCLUDE_DIRS} include)
    target_compile_definitions(CostForge PRIVATE COSTFORGE_LLVM_PASS ${LLVM_DEFINITIONS})
    target_compile_options(CostForge PRIVATE -fno-rtti)
    target_link_libraries(CostForge costforge_core)
    
    # Don't link LLVM libs — pass is loaded as plugin
    set_target_properties(CostForge PROPERTIES
        PREFIX ""
        SUFFIX ".so"
    )
endif()

# ── v0.3.0 tests ────────────────────────────────────

add_executable(test_v030 tests/test_v030.cpp)
target_link_libraries(test_v030 costforge_core)
add_test(NAME test_v030 COMMAND test_v030)

# ── v0.3.2 tests (LoopTiler + AutoTuner) ────────────

add_executable(test_v032 tests/test_v032.cpp)
target_link_libraries(test_v032 costforge_core)
add_test(NAME test_v032 COMMAND test_v032)
