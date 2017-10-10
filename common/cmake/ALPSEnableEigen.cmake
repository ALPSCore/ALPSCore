# enable using Eigen
option(ALPS_INSTALL_EIGEN "Download and install Eigen3 together with ALPSCore" OFF)
mark_as_advanced(ALPS_INSTALL_EIGEN)
set(ALPS_EIGEN_MIN_VERSION "3.3.4" CACHE STRING "Minimum Eigen version required by ALPSCore")
mark_as_advanced(ALPS_EIGEN_MIN_VERSION)

# Add eigen to the current module (target ${PROJECT_NAME})
function(add_eigen)
  message(STATUS "eigen requested")

  if (NOT ALPS_EIGEN_DOWNLOAD_DIR)
    set(ALPS_EIGEN_DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/eigen")
  endif()
  if (NOT ALPS_EIGEN_BUILD_DIR)
    set(ALPS_EIGEN_BUILD_DIR "${ALPS_EIGEN_DOWNLOAD_DIR}/000build")
  endif()

  if (NOT ALPS_INSTALL_EIGEN)
    find_package(Eigen3 ${ALPS_EIGEN_MIN_VERSION})
    if (NOT Eigen3_FOUND)
      message(FATAL_ERROR
" The required library Eigen3 has not been found on your system.
 Your could try the following options:
 1. Set environment variable Eigen3_DIR
    to point to the root of your CMake-based Eigen3 installation.
 2. Rerun CMake with option:
     -DEIGEN3_INCLUDE_DIR=<path to your Eigen3 directory> 
    to point to the location of Eigen3 headers.
 3. Rerun CMake with option:
     -DALPS_INSTALL_EIGEN=true 
    to request the installation script to attempt to download and install Eigen3.

    In the latter case, you may optionally also set:
     -DALPS_EIGEN_DOWNLOAD_DIR=<path to downloaded Eigen3> 
     (currently set to ${ALPS_EIGEN_DOWNLOAD_DIR})

     -DALPS_EIGEN_BUILD_DIR=<path where to set up Eigen3>
      (currently set to ${ALPS_EIGEN_BUILD_DIR})
")
    endif()

    if (NOT TARGET Eigen3::Eigen)
      message("DEBUG: Eigen3 target not found")
      add_library(Eigen3::Eigen INTERFACE IMPORTED)
      set_target_properties(Eigen3::Eigen PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${EIGEN3_INCLUDE_DIR})
    endif()
    target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen)
  else()
    message(FATAL_ERROR "Eigen installation is not yet implemented")
  endif()
endfunction()
