# 
# alps-utility cmake configuration script.
# 

cmake_minimum_required (VERSION 2.8)

# define project name
project (alps-utility CXX)
message ("\nALPSCore module : ${PROJECT_NAME}\n")
# add common modules from ../common/cmake
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../common/cmake)
include(ALPSCommonModuleDefinitions)

# Boost
#set(Boost_USE_STATIC_LIBS        ON)
#set(Boost_USE_STATIC_RUNTIME    OFF)
find_package (Boost COMPONENTS filesystem system REQUIRED)
    message(STATUS "Boost includes: " ${Boost_INCLUDE_DIRS} )
    message(STATUS "Boost libs: " ${Boost_LIBRARIES} )

# LINK_ALL - link for libraries
set (LINK_ALL ${Boost_LIBRARIES})

include_directories(
    ${Boost_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include
)

configure_file("${PROJECT_SOURCE_DIR}/include/config.h.in" "${PROJECT_BINARY_DIR}/include/alps/config.h")
install(FILES "${PROJECT_BINARY_DIR}/include/alps/config.h" DESTINATION include/alps) 

add_subdirectory(src)

# add tests - take two parameters Testing and TestXMLOutput
option(Testing "Enable testing" ON)
if (Testing)
    enable_testing()
    add_subdirectory(test)
endif (Testing)

# generate documentation
set(DOXYFILE_SOURCE_DIR "./include")
include(UseDoxygen)

# installation
install(DIRECTORY include DESTINATION ${CMAKE_INSTALL_PREFIX} 
          FILES_MATCHING PATTERN "*.hpp" PATTERN "*.hxx"
        )

# Generate pkg-config file
configure_file("${PROJECT_SOURCE_DIR}/${PROJECT_NAME}.pc.in" "${PROJECT_BINARY_DIR}/${PROJECT_NAME}.pc")
install(FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}.pc" DESTINATION "lib/pkgconfig")

# Generate cmake Find module
set(PROJECT_SEARCH_FILE alps/utility/data.hpp)
configure_file("${PROJECT_SOURCE_DIR}/../common/cmake/FindALPSModule.cmake.in" "${PROJECT_BINARY_DIR}/Find${PROJECT_NAME}.cmake" @ONLY)
install(FILES "${PROJECT_BINARY_DIR}/Find${PROJECT_NAME}.cmake" DESTINATION "share/cmake/Modules")

