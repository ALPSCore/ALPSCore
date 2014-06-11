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
    message(STATUS "Compiler does not support MPI. Specify mpicxx wrapper as CXX to enable MPI")
  endif()
else()
  message(STATUS "MPI disabled. Set ENABLE_MPI to ON to enable")
endif()
