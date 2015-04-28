#
# This cmake script enables MPI support in ALPSCore.
# It is done in the following way
# 1. Check the CXX compiler for MPI support.
# 2. If CXX doesn't support MPI - try to find the MPI compiler
# and check it for being the same compiler as CXX 
# 3. Otherwise disable MPI 
# 

# configurable option
option(ENABLE_MPI "Enable MPI build" ON)
set(ALPS_HAVE_MPI false)
if (ENABLE_MPI)
  # try compiling sample mpi code 
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
    # CXX Compiler doesn't support MPI - try to locate it
    message(STATUS "Compiler does not support MPI. Trying to find MPI") 

    find_package(MPI)
    if (${MPI_CXX_FOUND}) 

        # check that the versions of compilers are the same
        execute_process(COMMAND ${MPI_CXX_COMPILER}   "-dumpversion" OUTPUT_VARIABLE mpi_version OUTPUT_STRIP_TRAILING_WHITESPACE)
        execute_process(COMMAND ${CMAKE_CXX_COMPILER} "-dumpversion" OUTPUT_VARIABLE cxx_version OUTPUT_STRIP_TRAILING_WHITESPACE)
        if (${mpi_version} EQUAL ${cxx_version})
            set(ALPS_HAVE_MPI TRUE)
            message(STATUS "MPI : Using ${MPI_CXX_COMPILER}")
            list(APPEND CMAKE_CXX_FLAGS ${MPI_CXX_COMPILE_FLAGS}) 
            include_directories(${MPI_CXX_INCLUDE_PATH} ${MPI_C_INCLUDE_PATH})
            link_libraries(${MPI_CXX_LIBRARIES})
        else()
            message(WARNING "mpi compiler doesn't match the cxx compiler. Please specify mpicxx wrapper as CXX to enable MPI")
        endif()
    else()
        message(WARNING "MPI not found.")
    endif(${MPI_CXX_FOUND})
  endif()
else()
  message(STATUS "MPI disabled. Set ENABLE_MPI to ON to enable")
endif()
