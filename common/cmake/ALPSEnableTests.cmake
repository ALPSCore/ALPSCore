#
# This cmake script adds test to a project from the 'test' directory in the ALPS module
#

# check xml output
option(TestXMLOutput "Output tests to xml" OFF)

# default number of MPI processes for MPI testing
set(ALPS_TEST_MPI_NPROC 1 CACHE STRING "Default number of MPI processes to use for MPI-enabled tests")

# Find gtest or otherwise fetch it into the build_dir/gtest
function(UseGtest gtest_root)  
    #set (gtest_root ${ALPS_ROOT_DIR}/../common/deps/gtest-1.7.0)
    message(STATUS "gtest source specified at ${gtest_root}")
    find_path(gtest_root NAMES "include/gtest/gtest.h" HINTS ${gtest_root})

    message(STATUS "gtest is in ${gtest_root}")
    if (NOT TARGET ${gtest_root})
        # Hack to suppress all warnings in gtest
        set(save_cxx_flags_ ${CMAKE_CXX_FLAGS})
        set(save_c_flags_ ${CMAKE_C_FLAGS})
        # FIXME: this is actually gcc/clang/icc specific
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -w")
        add_subdirectory(${gtest_root} ${PROJECT_BINARY_DIR}/gtest)
        set(CMAKE_CXX_FLAGS ${save_cxx_flags_})
        set(CMAKE_C_FLAGS ${save_c_flags_})
    endif()
            
    # export gtest variables
    set (GTEST_ROOT ${gtest_root} PARENT_SCOPE)
    set (GTEST_INCLUDE_DIR ${gtest_root}/include PARENT_SCOPE)
    set (GTEST_MAIN_LIBRARIES gtest_main PARENT_SCOPE)
    set (GTEST_MAIN_LIBRARY gtest_main PARENT_SCOPE)
    set (GTEST_LIBRARY gtest PARENT_SCOPE)
endfunction()

# enable testing with gtest - fetch it if needed
if (NOT tests_are_already_enabled) 
    if (ALPS_GLOBAL_BUILD)
        set(gtest_root "${CMAKE_SOURCE_DIR}/common/deps/gtest-1.7.0")
    else(ALPS_GLOBAL_BUILD)
        set(gtest_root "${PROJECT_SOURCE_DIR}/../common/deps/gtest-1.7.0")
    endif(ALPS_GLOBAL_BUILD)

    UseGtest(${gtest_root})
    unset(gtest_root)
    find_package(GTest)
    # set (LINK_TEST  ${GTEST_MAIN_LIBRARIES}) 
    include_directories(SYSTEM ${GTEST_INCLUDE_DIRS})
    set(tests_are_already_enabled TRUE)
endif(NOT tests_are_already_enabled)


# internal function to build a test executable
# MAIN|NOMAIN TARGET <target_name> EXE <exe_name> SRCS <src1> <src2>...
function(alps_build_gtest_)
    include(CMakeParseArguments)
    cmake_parse_arguments("arg" "" "NOMAIN;TARGET" "SRCS" ${ARGN})
    set(usage_help_ "alps_build_gtest_([NOMAIN true|false] TARGET tgt SRCS file1 file2...)")
    if (arg_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR
            "Unknown parameters: ${arg_UNPARSED_ARGUMENTS}\n"
            "Usage: ${usage_help_}")
    endif()
    if (NOT arg_TARGET)
        message(FATAL_ERROR "'TARGET' must be specified in ${usage_help_}")
    endif()

    add_executable(${arg_TARGET} ${arg_SRCS})
    if (ALPS_BUILD_STATIC)
        set_property(TARGET ${arg_TARGET} PROPERTY LINK_SEARCH_START_STATIC 1)
        set_property(TARGET ${arg_TARGET} PROPERTY LINK_SEARCH_END_STATIC 1)
    endif()

    if (arg_NOMAIN)
        set(link_test_ ${GTEST_LIBRARY})
    else()
        set(link_test_ ${GTEST_MAIN_LIBRARIES})
    endif()
    
    target_link_libraries(${arg_TARGET} ${PROJECT_NAME} ${${PROJECT_NAME}_DEPENDS} ${link_test_})
endfunction()      
  

# custom function to add test with xml output and linked to gtest
# arg0 - test (assume the source is ${test}.cpp)
# optional arg: NOMAIN: do not link libgtest_main containing main()
# optional arg: MAIN: do link libgtest_main containing main()
# optional arg: PARTEST: run test in parallel using N processes, where N is run-time value of environment variable ALPS_TEST_MPI_NPROC
#               (or 1 if the variable is not set) (FIXME: make this a configurable constant!)
# optional arg: COMPILEFAIL: expect compilation failure as positive test outcome
# optional arg: SRCS source1 source2... : additional source files
# Affected by: ${PROJECT_NAME}_DEPENDS variable.
function(alps_add_gtest test)
    include(CMakeParseArguments)
    cmake_parse_arguments("arg" "NOMAIN;MAIN;PARTEST;COMPILEFAIL" "" "SRCS" ${ARGN})
    set(usage_help_ "alps_add_gtest(testname [MAIN|NOMAIN] [PARTEST] [COMPILEFAIL] [SRCS extra_sources...])")
    if (TestXMLOutput)
        set (test_xml_output_ --gtest_output=xml:${test}.xml)
    endif()
    if (arg_NOMAIN AND arg_MAIN)
        message(FATAL_ERROR "Incorrect use of ${usage_help_}")
    endif()
    if (arg_UNPARSED_ARGUMENTS) 
        message(FATAL_ERROR
            "Unknown parameters: ${arg_UNPARSED_ARGUMENTS}\n"
            "Usage: ${usage_help_}")
    endif()

    set(sources_ ${test} ${arg_SRCS})

    alps_build_gtest_(NOMAIN ${arg_NOMAIN} TARGET ${test} SRCS ${sources_})
    
    if (arg_COMPILEFAIL)
        set(nocompile_target_ ${test}.nocompile)
        alps_build_gtest_(NOMAIN ${arg_NOMAIN} TARGET ${nocompile_target_} SRCS ${sources_})
        set_property(TARGET ${nocompile_target_} PROPERTY EXCLUDE_FROM_ALL 1)
        target_compile_definitions(${nocompile_target_} PRIVATE "-DALPS_TEST_EXPECT_COMPILE_FAILURE")

        set(build_cmd_ "cmake --build . --target ${nocompile_target_}")
        set(run_fail_cmd_ "$<TARGET_FILE:${nocompile_target_}> ${test_xml_output_}")
        set(run_ok_cmd_ "$<TARGET_FILE:${test}> ${test_xml_output_}")
        set(shell_cmd_ "if ${build_cmd_}\; then ${run_fail_cmd_}\; false\; else ${run_ok_cmd_}\; fi")
        set(cmd_ "/bin/sh" "-c" "${shell_cmd_}")
    elseif (arg_PARTEST AND ALPS_HAVE_MPI) 
        # if unspecified, we assume the standard mpi launcher, according to MPI-3.1 specs, Chapter 8.8
        # (see https://www.mpi-forum.org/docs/mpi-3.1/mpi31-report/node228.htm#Node228)
        if (NOT DEFINED MPIEXEC) 
            set(MPIEXEC "mpiexec")
        endif()
        if (NOT DEFINED MPIEXEC_NUMPROC_FLAG)
            set(MPIEXEC_NUMPROC_FLAG "-n")
        endif()

        # we also allow test-time override of the MPI launcher and its arguments
        # (NOTE: POSIX shell is assumed)
        set(cmd_ "/bin/sh" "-c" "\${ALPS_TEST_MPIEXEC:-${MPIEXEC}} \${ALPS_TEST_MPI_NPROC_FLAG:-${MPIEXEC_NUMPROC_FLAG}} \${ALPS_TEST_MPI_NPROC:-${ALPS_TEST_MPI_NPROC}} \${ALPS_TEST_MPIEXEC_PREFLAGS:-${MPIEXEC_PREFLAGS}} $<TARGET_FILE:${test}> \${ALPS_TEST_MPIEXEC_POSTFLAGS:-${MPIEXEC_POSTFLAGS}} ${test_xml_output_}")
    else()
        set(cmd_ $<TARGET_FILE:${test}> ${test_xml_output_})
    endif()
    add_test(NAME ${test} COMMAND ${cmd_})
endfunction(alps_add_gtest)
