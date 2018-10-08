# enable using Eigen
option(ALPS_INSTALL_EIGEN "Download and install Eigen3 together with ALPSCore" OFF)
mark_as_advanced(ALPS_INSTALL_EIGEN)
set(ALPS_EIGEN_MIN_VERSION "3.3.4" CACHE STRING "Minimum Eigen version required by ALPSCore")
mark_as_advanced(ALPS_EIGEN_MIN_VERSION)

set(ALPS_EIGEN_DOWNLOAD_LOCATION "https://bitbucket.org/eigen/eigen/get/${ALPS_EIGEN_MIN_VERSION}.tar.gz"
    CACHE STRING "Eigen3 download location")
mark_as_advanced(ALPS_EIGEN_DOWNLOAD_LOCATION)

set(ALPS_EIGEN_DOWNLOAD_METHODS cmake wget curl CACHE STRING "List of Eigen3 download methods to attempt")
mark_as_advanced(ALPS_EIGEN_DOWNLOAD_METHODS)
if (NOT ALPS_EIGEN_DOWNLOAD_METHODS)
  message(FATAL_ERROR "The ALPS_EIGEN_DOWNLOAD_METHODS variable cannot be empty")
endif()

set(ALPS_BUNDLE_DOWNLOAD_TRIES 1 CACHE STRING "How many times to attempt a download of a dependency")
mark_as_advanced(ALPS_BUNDLE_DOWNLOAD_TRIES)


# Function to download a resource using CMake file() built-in
# Arguments:
#  url      : URL to download
#  destfile : where to download to
#  ntries   : how many times to try
#  statvar  : name of the variable to store status (0 == Success)
function(cmake_download_ url destfile ntries statvar)
  foreach(loop_var RANGE ${ntries})
    file(DOWNLOAD ${url} ${destfile}
      INACTIVITY_TIMEOUT 60
      TIMEOUT 600
      STATUS status
      SHOW_PROGRESS)
    if (status EQUAL 0)
      break()
    endif()
  endforeach()
  set(${statvar} ${status} PARENT_SCOPE)
endfunction()

# Function to download a resource using wget
# Arguments:
#  url      : URL to download
#  destfile : where to download to
#  ntries   : how many times to try
#  statvar  : name of the variable to store status (0 == Success)
function(wget_download_ url destfile ntries statvar)
  find_program(WGET_BINARY wget DOC "Location of wget utility")
  mark_as_advanced(WGET_BINARY)
  if (NOT WGET_BINARY)
    set(status, "Cannot find wget")
  else()
    get_filename_component(dest_dir ${destfile} DIRECTORY)
    file(MAKE_DIRECTORY ${dest_dir})
    foreach(loop_var RANGE ${ntries})
      execute_process(COMMAND ${WGET_BINARY} "-O" ${destfile} ${url} RESULT_VARIABLE status TIMEOUT 600)
      if (status EQUAL 0)
        break()
      endif()
    endforeach()
  endif()
  set(${statvar} ${status} PARENT_SCOPE)
endfunction()

# Function to download a resource using curl
# Arguments:
#  url      : URL to download
#  destfile : where to download to
#  ntries   : how many times to try
#  statvar  : name of the variable to store status (0 == Success)
function(curl_download_ url destfile ntries statvar)
  find_program(CURL_BINARY curl DOC "Location of curl utility")
  mark_as_advanced(CURL_BINARY)
  if (NOT CURL_BINARY)
    set(status, "Cannot find curl")
  else()
    get_filename_component(dest_dir ${destfile} DIRECTORY)
    file(MAKE_DIRECTORY ${dest_dir})
    foreach(loop_var RANGE ${ntries})
      execute_process(COMMAND ${CURL_BINARY} "--output" ${destfile} ${url} RESULT_VARIABLE status TIMEOUT 600)
      if (status EQUAL 0)
        break()
      endif()
    endforeach()
  endif()
  set(${statvar} ${status} PARENT_SCOPE)
endfunction()

# Check if the gzip file is readable and in proper format (e.g., not empty)
# Arguments:
#   filename : name of the file to check
#   ok_var   : name of the variable to return the status (TRUE if the file is OK)
function(check_gz_ filename ok_var)
  set(sig_ "")
  file(READ "${filename}" sig_ LIMIT 3 HEX)
  if (sig_ STREQUAL "1f8b08")
    set(ok TRUE)
  else()
    set(ok "")
  endif()
  set(${ok_var} ${ok} PARENT_SCOPE)
endfunction()

# Add eigen to the current module (target ${PROJECT_NAME})
# Sets EIGEN3_VERSION variable in the parent scope
function(add_eigen)
  message(STATUS "eigen requested")

  if (NOT ALPS_EIGEN_UNPACK_DIR)
    set(ALPS_EIGEN_UNPACK_DIR "${CMAKE_BINARY_DIR}/eigen")
  endif()
  if (NOT ALPS_EIGEN_TGZ_FILE)
    set(ALPS_EIGEN_TGZ_FILE "${ALPS_EIGEN_UNPACK_DIR}/eigen.tgz")
  endif()

  # CAUTION, the message contains significant spaces in seemingly-empty lines.
  set(eigen_install_options_msg "
 1. Set environment variable Eigen3_DIR
    to point to the root of your CMake-based Eigen3 installation.
 2. Rerun CMake with option:
     -DEIGEN3_INCLUDE_DIR=<path to your Eigen3 directory> 
    to point to the location of Eigen3 headers.
 3. Rerun CMake with option:
     -DALPS_INSTALL_EIGEN=true 
    to request the installation script to attempt to download Eigen3
    and co-install it with ALPSCore. (If it still fails, remove your
    CMakeCache.txt and try again).
 
    In the latter case, you may optionally also set:
     -DALPS_EIGEN_UNPACK_DIR=<path to directory to unpack Eigen> 
     (currently set to ${ALPS_EIGEN_UNPACK_DIR})
 
     -DALPS_EIGEN_TGZ_FILE=<path to Eigen3 archive file>
     (currently set to ${ALPS_EIGEN_TGZ_FILE})
 
     -DEIGEN3_INSTALL_DIR=<path to unpacked Eigen3>
     (if you want to co-install a specific version of Eigen)
")

  # CAUTION, the message contains significant spaces in seemingly-empty lines.
  set(eigen_download_options_msg "
 If you wish to co-install Eigen3, please do the the following:
 1) Download the file manually from ${ALPS_EIGEN_DOWNLOAD_LOCATION} ;
 2) Set CMake variable ALPS_EIGEN_TGZ_FILE to the location of the downloaded file.
 Alternatively, you can download and unpack Eigen3 in the directory of your choice and set CMake variable
 EIGEN3_INCLUDE_DIR to the location of the directory. Note that Eigen3 will still be co-installed with ALPSCore
 unless you explicitly set CMake variable ALPS_INSTALL_EIGEN to FALSE.")
  
  if (NOT ALPS_INSTALL_EIGEN)
    find_package(Eigen3 ${ALPS_EIGEN_MIN_VERSION})
    if (NOT EIGEN3_FOUND AND NOT DEFINED EIGEN3_VERSION_OK) # CMake 3.3+ would use Eigen3_FOUND
      message(FATAL_ERROR
" 
 The required library Eigen3 has not been found on your system.
 Your could try the following options:${eigen_install_options_msg}")
    elseif(NOT EIGEN3_FOUND AND DEFINED EIGEN3_VERSION_OK)
       message(FATAL_ERROR
" 
 The Eigen3 library has been found at ${EIGEN3_INCLUDE_DIR} on your system;
 HOWEVER, your version is ${EIGEN3_VERSION} 
 which is less than the required version ${ALPS_EIGEN_MIN_VERSION}.
 Please try upgrading your installation of the Eigen3 library
 or use a different installation; in the latter case, the following
 options are available: ${eigen_install_options_msg}")
    endif()

    target_include_directories(${PROJECT_NAME} PUBLIC ${EIGEN3_INCLUDE_DIR})
    message(STATUS "Using Eigen3 installed at ${EIGEN3_INCLUDE_DIR}")

  else(NOT ALPS_INSTALL_EIGEN)

    message(STATUS "Eigen co-installation requested")
    
    set(untar_cmd_ "${CMAKE_COMMAND};-E;tar;xvz")

    if (NOT EIGEN3_INCLUDE_DIR)
      message(STATUS "Trying to download and unpack Eigen3")
      if (NOT EXISTS "${ALPS_EIGEN_TGZ_FILE}")
        foreach(method_ ${ALPS_EIGEN_DOWNLOAD_METHODS})
          message(STATUS "Downloading Eigen3 via ${method_}, timeout 600 sec")
          if (method_ STREQUAL cmake)
            cmake_download_(${ALPS_EIGEN_DOWNLOAD_LOCATION} ${ALPS_EIGEN_TGZ_FILE} ${ALPS_BUNDLE_DOWNLOAD_TRIES} status_)
          elseif (method_ STREQUAL wget)
            wget_download_(${ALPS_EIGEN_DOWNLOAD_LOCATION} ${ALPS_EIGEN_TGZ_FILE} ${ALPS_BUNDLE_DOWNLOAD_TRIES} status_)
          elseif (method_ STREQUAL curl)
            curl_download_(${ALPS_EIGEN_DOWNLOAD_LOCATION} ${ALPS_EIGEN_TGZ_FILE} ${ALPS_BUNDLE_DOWNLOAD_TRIES} status_)
          else()
            message(FATAL_ERROR "Unknown download method ${method_}")
          endif()
          if (status_ EQUAL 0)
            message(STATUS "Downloaded successfully via ${method_}")
            break()
          else()
            message("Failed to download via ${method_}, trying the next one, if any")
          endif()
        endforeach()
        check_gz_("${ALPS_EIGEN_TGZ_FILE}" ok_)
        if (NOT status_ EQUAL 0 OR NOT ok_)
          file(REMOVE "${ALPS_EIGEN_TGZ_FILE}")
          message(FATAL_ERROR "Failed to download ${ALPS_EIGEN_TGZ_FILE} "
            "from ${ALPS_EIGEN_DOWNLOAD_LOCATION}: "
            "status=" ${status_} ${eigen_download_options_msg}
            )
        endif()
      else()
        message(STATUS "File ${ALPS_EIGEN_TGZ_FILE} is already downloaded")
        check_gz_("${ALPS_EIGEN_TGZ_FILE}" ok_)
        if (NOT ok_)
          message(FATAL_ERROR "File ${ALPS_EIGEN_TGZ_FILE} is empty, corrupted, or unreadable."
            " Please remove or replace the file before you try again." ${eigen_download_options_msg})
        endif()
      endif()

      set(unpack_subdir_ "${ALPS_EIGEN_UNPACK_DIR}/000unpack")
      file(MAKE_DIRECTORY "${unpack_subdir_}")
      file(REMOVE_RECURSE "${unpack_subdir_}")
      file(MAKE_DIRECTORY "${unpack_subdir_}")
      execute_process(
        COMMAND ${untar_cmd_} ${ALPS_EIGEN_TGZ_FILE}
        WORKING_DIRECTORY "${unpack_subdir_}"
        RESULT_VARIABLE status_)
      if (NOT status_ EQUAL 0)
        message(FATAL_ERROR "Cannot unpack the file ${ALPS_EIGEN_TGZ_FILE}: "
          "status=" ${status_}
          " 
 It is possible that download was unsuccessful. Please remove the file, verify the download location, and try again.")
      endif()
      message(STATUS "Unpacked successfully")

      file(GLOB_RECURSE sigfile_ "${unpack_subdir_}/signature_of_eigen3_matrix_library")
      if (NOT sigfile_)
        message(FATAL_ERROR "Cannot find Eigen3 in the unpacked archive under ${unpack_subdir_}")
      endif()
      unset(EIGEN3_INCLUDE_DIR)
      unset(EIGEN3_INCLUDE_DIR CACHE)
      get_filename_component(EIGEN3_INCLUDE_DIR "${sigfile_}" DIRECTORY CACHE)
    endif(NOT EIGEN3_INCLUDE_DIR)

    # Here EIGEN3_INCLUDE_DIR is set, find_package() will only check the version 
    message(STATUS "Searching for Eigen3 in ${EIGEN3_INCLUDE_DIR}")
    find_package(Eigen3 ${ALPS_EIGEN_MIN_VERSION})
    if (NOT EIGEN3_FOUND) # CMake 3.3+ would use Eigen3_FOUND
      message(FATAL_ERROR
        "\nCannot find suitable Eigen3 in ${EIGEN3_INCLUDE_DIR}."
        " Make sure that Eigen3 is indeed at the specified location, "
        "or unset the variable `EIGEN3_INCLUDE_DIR` to download Eigen3 "
        "to your machine.")
    endif()

    set(eigen_install_dir_ ${CMAKE_INSTALL_PREFIX}/deps/eigen)
    target_include_directories(${PROJECT_NAME} PUBLIC
      $<BUILD_INTERFACE:${EIGEN3_INCLUDE_DIR}>
      $<INSTALL_INTERFACE:${eigen_install_dir_}>)
    install(DIRECTORY "${EIGEN3_INCLUDE_DIR}/Eigen" "${EIGEN3_INCLUDE_DIR}/unsupported" DESTINATION ${eigen_install_dir_})
    
  endif(NOT ALPS_INSTALL_EIGEN)

  # assertion
  if (NOT EIGEN3_FOUND OR NOT EIGEN3_VERSION) # CMake 3.3+ would use Eigen3_FOUND
    message(FATAL_ERROR "Assertion error: Eigen3 must have been found and versioned. "
      "\nEIGEN3_FOUND=${EIGEN3_FOUND}"
      "\nEIGEN3_VERSION=${EIGEN3_VERSION}"
      "\nPlease report this to ALPSCore developers.")
  endif()
  set(EIGEN3_VERSION ${EIGEN3_VERSION} PARENT_SCOPE)
endfunction()
