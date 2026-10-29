cmake_minimum_required(VERSION 3.12)
project(madmom_native LANGUAGES CXX)

add_library(madmom_native SHARED src/online.cpp src/nn.cpp src/decoders.cpp)
target_compile_definitions(madmom_native PRIVATE MADMOM_NATIVE_BUILD=1)
target_include_directories(madmom_native PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_compile_features(madmom_native PRIVATE cxx_std_17)
# Reassociation and contraction change recurrent model results.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(madmom_native PRIVATE -ffp-contract=off)
elseif(MSVC)
    target_compile_options(madmom_native PRIVATE /fp:strict)
endif()
set_target_properties(madmom_native PROPERTIES VERSION 1.0 SOVERSION 1)
install(TARGETS madmom_native LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(FILES include/madmom_engine.h include/madmom_exports.h
    include/madmom_native.h include/madmom_nn.h include/madmom_decoders.h
    DESTINATION include)

include(CTest)
set(MADMOM_NATIVE_TEST_ASSET "" CACHE FILEPATH "Exported model for native API tests")
if(BUILD_TESTING AND MADMOM_NATIVE_TEST_ASSET)
    add_executable(madmom_native_api_test tests/api.cpp)
    target_compile_features(madmom_native_api_test PRIVATE cxx_std_11)
    target_link_libraries(madmom_native_api_test PRIVATE madmom_native)
    add_test(NAME native_api COMMAND madmom_native_api_test
        "${MADMOM_NATIVE_TEST_ASSET}")
endif()
