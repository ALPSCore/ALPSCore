#
# This cmake script adds test to a project from the 'test' directory in the ALPS module
#

# check xml output
option(TestXMLOutput "Output tests to xml" OFF)

# enable testing with gtest
if (NOT tests_are_already_enabled) 
    message(STATUS "Building tests")
    #set (LINK_TEST gtest_main)
    #add_subdirectory(${gtest_ROOT} ${PROJECT_BINARY_DIR}/gtest)
    #include_directories(${gtest_INCLUDE_DIR})
    find_package(GTest QUIET)

    if (NOT GTEST_FOUND) 
        include(UseGtest) # fetch gtest
        find_package(GTest)
    endif (NOT GTEST_FOUND) 

    set (LINK_TEST  ${GTEST_MAIN_LIBRARIES}) 
    include_directories(${GTEST_INCLUDE_DIRS})
    set(tests_are_already_enabled TRUE)
endif(NOT tests_are_already_enabled)

# custom function to add test with xml output and linked to gtest
# arg0 - test (assume the source is ${test}.cpp
function(alps_add_gtest test)
    if (TestXMLOutput)
        set (test_xml_output --gtest_output=xml:${test}.xml)
    endif(TestXMLOutput)

    if(${ARGC} EQUAL 2)
        set(source "${ARGV1}/${test}.cpp")
    else(${ARGC} EQUAL 2)
        set(source "${test}.cpp")
    endif(${ARGC} EQUAL 2)

    add_executable(${test} ${source})
    target_link_libraries(${test} ${PROJECT_NAME} ${LINK_ALL} ${LINK_TEST})
    add_test(NAME ${test} COMMAND ${test} ${test_xml_output})
endfunction(alps_add_gtest)
