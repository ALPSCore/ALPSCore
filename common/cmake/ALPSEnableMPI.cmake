#
# This cmake script enables MPI support in ALPSCore.
# It is done in the following way
# 1. Check the CXX compiler for MPI support.
# 2. If CXX doesn't support MPI - try to find the MPI compiler,
#    check it for being the same compiler as CXX,
#    and add libraries and headers to the target ${PROJECT_NAME}
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
        message(STATUS "MPI : Found compiler ${MPI_CXX_COMPILER}")
        # check that the versions of compilers are the same
        execute_process(COMMAND ${MPI_CXX_COMPILER}   "-dumpversion" OUTPUT_VARIABLE mpi_version OUTPUT_STRIP_TRAILING_WHITESPACE)
        execute_process(COMMAND ${CMAKE_CXX_COMPILER} "-dumpversion" OUTPUT_VARIABLE cxx_version OUTPUT_STRIP_TRAILING_WHITESPACE)
        if (NOT cxx_version)
          set(cxx_version ${CMAKE_CXX_COMPILER_VERSION})
        endif()
        if (NOT mpi_version OR NOT cxx_version OR NOT mpi_version EQUAL cxx_version)
            message(WARNING "MPI compiler doesn't match the C++ compiler.
MPI compiler is: ${MPI_CXX_COMPILER}
MPI compiler version is: ${mpi_version}
C++ compiler is: ${CMAKE_CXX_COMPILER}
C++ compiler version is: ${cxx_version}
Depending on your platform this may lead to problems.")
        endif()
        set(ALPS_HAVE_MPI TRUE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_CXX_COMPILE_FLAGS}") 
        message(STATUS "MPI : Using ${CMAKE_CXX_COMPILER}")
        if (MPI_CXX_COMPILE_FLAGS)	
            message(STATUS "MPI : with options ${MPI_CXX_COMPILE_FLAGS}")
        endif()
    else()
        message(WARNING "MPI not found.")
    endif(${MPI_CXX_FOUND})
  endif()
else()
  message(STATUS "MPI disabled. Set ENABLE_MPI to ON to enable")
endif()
