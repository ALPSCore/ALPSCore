# 
# enables testing with google test
# provides add_gtest(test) and add_gtest_release(test) commands
#

# check xml output
option(TestXMLOutput "Output tests to xml" OFF)

# Find gtest or otherwise fetch it into the build_dir/gtest
# Adapted from alpscore
function(UseGtest)  
    set (gtest_root ${ARGN})
    if (DEFINED gtest_root)
        message(STATUS "gtest source specified at ${gtest_root}")
        find_path(gtest_root NAMES "include/gtest/gtest.h" HINTS ${gtest_root})
        if (NOT IS_DIRECTORY ${gtest_root})
            message(WARNING "Provided wrong GTEST_ROOT. Please unset GTEST_ROOT - gtest will be fetched")
            unset(gtest_root CACHE)
        endif()
    endif()

    if (NOT DEFINED gtest_root)
        find_path(gtest_root1 NAMES "include/gtest/gtest.h" HINTS ${CMAKE_SOURCE_DIR}/gtest-1.6.0  ${CMAKE_SOURCE_DIR}/gtest-1.7.0  ${CMAKE_SOURCE_DIR}/gtest   ${CMAKE_BINARY_DIR}/gtest)
        if (IS_DIRECTORY ${gtest_root1})
            set (gtest_root ${gtest_root1})
        else()
            message(STATUS "Trying to fetch gtest via subversion")
            find_package(Subversion)
            execute_process(COMMAND "${Subversion_SVN_EXECUTABLE}" "checkout" "http://googletest.googlecode.com/svn/trunk/" "gtest" WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
            set (gtest_root "${CMAKE_BINARY_DIR}/gtest")
        endif()
        set (gtest_root1)
    endif()

    message(STATUS "gtest is in ${gtest_root}")
    if (NOT TARGET ${gtest_root})
        add_subdirectory(${gtest_root} ${PROJECT_BINARY_DIR}/gtest)
    endif()
            
    # export gtest variables
    set (GTEST_ROOT ${gtest_root} PARENT_SCOPE)
    set (GTEST_INCLUDE_DIR ${gtest_root}/include PARENT_SCOPE)
    set (GTEST_MAIN_LIBRARIES gtest_main PARENT_SCOPE)
    set (GTEST_MAIN_LIBRARY gtest_main PARENT_SCOPE)
    set (GTEST_LIBRARY gtest_main PARENT_SCOPE)
endfunction()

# custom function to add gtest with xml output
# arg0 - test (assume the source is ${test}.cpp
function(add_gtest test)
    if (TestXMLOutput)
        set (test_xml_output --gtest_output=xml:${test}.xml)
    endif(TestXMLOutput)

    if(${ARGC} EQUAL 2)
        set(source "${ARGV1}/${test}.cpp")
    else(${ARGC} EQUAL 2)
        set(source "${test}.cpp")
    endif(${ARGC} EQUAL 2)

    add_executable(${test} ${source})
    target_link_libraries(${test} ${LINK_ALL} ${LINK_TEST})
    add_test(NAME ${test} COMMAND ${test} ${test_xml_output})
endfunction(add_gtest)

# add test in release mode with gtest
function(add_gtest_release test)
    if (CMAKE_BUILD_TYPE STREQUAL Release)
        add_gtest(${test})
    endif()
endfunction(add_gtest_release)


# enable testing with gtest - fetch it if needed
if (NOT tests_are_already_enabled) 
    find_package(GTest QUIET)

    if (NOT GTEST_FOUND) 
        #include(UseGtest) # fetch gtest
        UseGtest(${GTEST_ROOT})
        find_package(GTest)
    endif (NOT GTEST_FOUND) 

    set (LINK_TEST  ${GTEST_MAIN_LIBRARIES}) 
    include_directories(${GTEST_INCLUDE_DIRS})
    set(tests_are_already_enabled TRUE)
endif(NOT tests_are_already_enabled)

