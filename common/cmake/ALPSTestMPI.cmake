#
# This cmake script checks the CXX compiler for MPI support
#
option(ENABLE_MPI ON)
if (ENABLE_MPI)
  set(scratch_directory ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY})
  set(test_file ${scratch_directory}/cmake_mpi_test.cpp)
  file(WRITE ${test_file}
      "#include <mpi.h>\n"
      "int main(int argc, char **argv) {\n"
      "  MPI_Init(&argc, &argv);\n"
      "  MPI_Finalize();\n"
      "}\n")
  try_compile(compiler_has_mpi ${scratch_directory} ${test_file})
  if (compiler_has_mpi)
    set(ALPS_HAVE_MPI TRUE)
    message(STATUS "Compiler supports MPI." ${CMAKE_CXX_COMPILER})
  else()
    message(STATUS "Compiler does not support MPI. Trying to find MPI") 

    find_package(MPI REQUIRED)

    set(mpi_is_ok false)
    # check that the versions of compilers are the same
    execute_process(COMMAND ${MPI_CXX_COMPILER}   "-dumpversion" OUTPUT_VARIABLE mpi_version OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} "-dumpversion" OUTPUT_VARIABLE cxx_version OUTPUT_STRIP_TRAILING_WHITESPACE)
    if (${mpi_version} EQUAL ${cxx_version})
        set(mpi_is_ok true)
    endif()

    if (mpi_is_ok)
        message(STATUS "MPI : Using ${MPI_CXX_COMPILER}")
        list(APPEND CMAKE_CXX_FLAGS ${MPI_CXX_COMPILE_FLAGS}) 
        include_directories(${MPI_CXX_INCLUDE_PATH} ${MPI_C_INCLUDE_PATH})
        link_libraries(${MPI_CXX_LIBRARIES})

    else(mpi_is_ok)
        message(FATAL_ERROR "mpi compiler doesn't match the cxx compiler. Please specify mpicxx wrapper as CXX to enable MPI")
    endif(mpi_is_ok)

  endif()
else()
  message(STATUS "MPI disabled. Set ENABLE_MPI to ON to enable")
endif()
