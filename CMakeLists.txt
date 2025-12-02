cmake_minimum_required(VERSION 3.16)

project(CatDogClassifier LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

add_library(catdog_lib STATIC
    src/catdog.c
)

target_include_directories(catdog_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

if(UNIX AND NOT APPLE)
    target_link_libraries(catdog_lib PUBLIC m)
elseif(APPLE)
    target_link_libraries(catdog_lib PUBLIC m)
endif()

add_executable(catdog src/main.c)
target_link_libraries(catdog PRIVATE catdog_lib)

add_executable(catdog_infer src/infer.c)
target_link_libraries(catdog_infer PRIVATE catdog_lib)

option(CATDOG_BUILD_TESTS "Build the lightweight Cat/Dog unit tests" ON)

if(CATDOG_BUILD_TESTS)
    include(CTest)

    add_executable(catdog_tests tests/test_network.c)
    target_link_libraries(catdog_tests PRIVATE catdog_lib)

    add_test(
        NAME catdog_unit_tests
        COMMAND catdog_tests
    )
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
