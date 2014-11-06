#  Python settings : 
#
#  This module checks that : 
#  - the python interpreter is working and version >= 2.6
#  - it has modules : distutils, numpy, tables, scipy
# 
#  This module defines the variables
#  - PYTHON_INTERPRETER : name of the python interpreter
#  - PYTHON_INCLUDE_DIRS : include for compilation
#  - PYTHON_NUMPY_INCLUDE_DIR : include for compilation with numpy
#  - PYTHON_LIBRARY : link flags 
#  - PYTHON_SITE_PKG : path to the standard packages of the python interpreter
#  - PYTHON_EXTRA_LIBS :  libraries which must be linked in when embedding
#  - PYTHON_LINK_FOR_SHARED :  linking flags needed when building a shared lib for external modules

if (NOT PYTHON_INTERPRETER)
  find_program(PYTHON_INTERPRETER python PATHS $ENV{PATH})
  if (NOT PYTHON_INTERPRETER)
    set (PYTHON_FOUND FALSE)
  else(NOT PYTHON_INTERPRETER)
    set(PYTHON_FOUND TRUE)
  endif(NOT PYTHON_INTERPRETER)
else (NOT PYTHON_INTERPRETER)
  set(PYTHON_FOUND TRUE)
endif (NOT PYTHON_INTERPRETER)

IF (PYTHON_FOUND)

  set(PYTHON_MINIMAL_VERSION 2.6)

  MESSAGE (STATUS "Python interpreter ${PYTHON_INTERPRETER}")
  #
  # The function EXEC_PYTHON_SCRIPT executes the_script in  python interpreter
  # and set the variable of output_var_name in the calling scope
  #
  FUNCTION ( EXEC_PYTHON_SCRIPT the_script output_var_name)
    IF ("${PYTHON_INTERPRETER}" MATCHES ".*ipython.*")
      EXECUTE_PROCESS(COMMAND ${PYTHON_INTERPRETER} "--c=${the_script}" 
        OUTPUT_VARIABLE res RESULT_VARIABLE returncode OUTPUT_STRIP_TRAILING_WHITESPACE)
    ELSE ("${PYTHON_INTERPRETER}" MATCHES ".*ipython.*")
      EXECUTE_PROCESS(COMMAND ${PYTHON_INTERPRETER} -c "${the_script}" 
        OUTPUT_VARIABLE res RESULT_VARIABLE returncode OUTPUT_STRIP_TRAILING_WHITESPACE)
    ENDIF ("${PYTHON_INTERPRETER}" MATCHES ".*ipython.*")
    IF (NOT returncode EQUAL 0)
      MESSAGE(FATAL_ERROR "The script : ${the_script} \n did not run properly in the Python interpreter. Check your python installation.") 
    ENDIF (NOT returncode EQUAL 0)
    SET( ${output_var_name} ${res} PARENT_SCOPE)
  ENDFUNCTION (EXEC_PYTHON_SCRIPT)

  #
  # Check the interpreter and its version
  #
  EXEC_PYTHON_SCRIPT ("import sys, string; print sys.version.split()[0]" PYTHON_VERSION)
  STRING(COMPARE GREATER ${PYTHON_MINIMAL_VERSION} ${PYTHON_VERSION} PYTHON_VERSION_NOT_OK)
  IF (PYTHON_VERSION_NOT_OK)
    MESSAGE(WARNING "Python intepreter version is ${PYTHON_VERSION} . It should be >= ${PYTHON_MINIMAL_VERSION}")
    SET(PYTHON_FOUND FALSE)
  ENDIF (PYTHON_VERSION_NOT_OK)
ENDIF (PYTHON_FOUND)

IF (PYTHON_FOUND)
  EXEC_PYTHON_SCRIPT ("import distutils " nulle) # check that distutils is there...
  EXEC_PYTHON_SCRIPT ("import numpy" nulle) # check that numpy is there...
  #EXEC_PYTHON_SCRIPT ("import scipy" nulle) # check that scipy is there...
  #EXEC_PYTHON_SCRIPT ("import tables" nulle) # check that tables is there...
  MESSAGE(STATUS "Python interpreter ok : version ${PYTHON_VERSION}" )

  #
  # Check for Python include path
  #
  EXEC_PYTHON_SCRIPT ("import distutils ; from distutils.sysconfig import * ; print distutils.sysconfig.get_python_inc()"  PYTHON_INCLUDE_DIRS )
  message(STATUS "PYTHON_INCLUDE_DIRS =  ${PYTHON_INCLUDE_DIRS}" )
  mark_as_advanced(PYTHON_INCLUDE_DIRS)
  FIND_PATH(TEST_PYTHON_INCLUDE patchlevel.h PATHS ${PYTHON_INCLUDE_DIRS} NO_DEFAULT_PATH)
  if (NOT TEST_PYTHON_INCLUDE)
    message (ERROR "The Python header files have not been found. Please check that you installed the Python headers and not only the interpreter.")
  endif (NOT TEST_PYTHON_INCLUDE)

  #
  # include files for numpy
  #
  EXEC_PYTHON_SCRIPT ("import numpy;print numpy.get_include()" PYTHON_NUMPY_INCLUDE_DIR)
  MESSAGE(STATUS "PYTHON_NUMPY_INCLUDE_DIR = ${PYTHON_NUMPY_INCLUDE_DIR}" )
  mark_as_advanced(PYTHON_NUMPY_INCLUDE_DIR)

  #
  # Check for site packages
  #
  EXEC_PYTHON_SCRIPT ("from distutils.sysconfig import * ;print get_python_lib(0,0)"
              PYTHON_SITE_PKG)
  MESSAGE(STATUS "PYTHON_SITE_PKG = ${PYTHON_SITE_PKG}" )
  mark_as_advanced(PYTHON_SITE_PKG)

    if (NOT WIN32)
      #
      # Check for Python library path
      #
      #EXEC_PYTHON_SCRIPT ("import string; from distutils.sysconfig import * ;print string.join(get_config_vars('VERSION'))"  PYTHON_VERSION_MAJOR_MINOR)         
      EXEC_PYTHON_SCRIPT ("import string; from distutils.sysconfig import *; print '%s/config' % get_python_lib(0,1)" PYTHON_LIBRARY_BASE_PATH)
      EXEC_PYTHON_SCRIPT ("import string; from distutils.sysconfig import *; print 'libpython%s' % string.join(get_config_vars('VERSION'))" PYTHON_LIBRARY_BASE_FILE)
      IF(BUILD_SHARED_LIBS)
        FIND_FILE(PYTHON_LIBRARY NAMES "${PYTHON_LIBRARY_BASE_FILE}.so" PATHS ${PYTHON_LIBRARY_BASE_PATH})
        IF(NOT PYTHON_LIBRARY)
          FIND_FILE(PYTHON_LIBRARY NAMES "${PYTHON_LIBRARY_BASE_FILE}.a" PATHS ${PYTHON_LIBRARY_BASE_PATH})
        ENDIF(NOT PYTHON_LIBRARY)
      ELSE(BUILD_SHARED_LIBS)
        FIND_FILE(PYTHON_LIBRARY NAMES "${PYTHON_LIBRARY_BASE_FILE}.a" PATHS ${PYTHON_LIBRARY_BASE_PATH})
      ENDIF(BUILD_SHARED_LIBS)
      IF(NOT PYTHON_LIBRARY)
        # On Debian/Ubuntu system, libpython*.so is located in /usr/lib/`gcc -print-multiarch`
        execute_process(COMMAND gcc -print-multiarch OUTPUT_VARIABLE TRIPLES)
        STRING(REGEX REPLACE "\n" "" TRIPLES ${TRIPLES})
        FIND_FILE(PYTHON_LIBRARY NAMES "${PYTHON_LIBRARY_BASE_FILE}.so" PATHS "/usr/lib/${TRIPLES}")
        IF(NOT PYTHON_LIBRARY)
          FIND_FILE(PYTHON_LIBRARY NAMES "${PYTHON_LIBRARY_BASE_FILE}.a" PATHS "/usr/lib/${TRIPLES}")
        ENDIF(NOT PYTHON_LIBRARY)
      ENDIF(NOT PYTHON_LIBRARY)
      MESSAGE(STATUS "PYTHON_LIBRARY = ${PYTHON_LIBRARY}" )
      mark_as_advanced(PYTHON_LIBRARY)

      #
      # libraries which must be linked in when embedding
      #
      EXEC_PYTHON_SCRIPT ("from distutils.sysconfig import * ;print (str(get_config_var('LOCALMODLIBS')) + ' ' + str(get_config_var('LIBS'))).strip()"
                  PYTHON_EXTRA_LIBS)
      MESSAGE(STATUS "PYTHON_EXTRA_LIBS =${PYTHON_EXTRA_LIBS}" )
      mark_as_advanced(PYTHON_EXTRA_LIBS)

      #
      # linking flags needed when embedding (building a shared lib)
      # To BE RETESTED
      #
      EXEC_PYTHON_SCRIPT ("from distutils.sysconfig import *;print get_config_var('LINKFORSHARED')"
                  PYTHON_LINK_FOR_SHARED)
      MESSAGE(STATUS "PYTHON_LINK_FOR_SHARED =  ${PYTHON_LINK_FOR_SHARED}" )
      mark_as_advanced(PYTHON_LINK_FOR_SHARED)
    endif(NOT WIN32)

  # Correction on Mac
  IF(APPLE)
      SET (PYTHON_LINK_FOR_SHARED -u _PyMac_Error -framework Python)
      SET (PYTHON_LINK_MODULE -bundle -undefined dynamic_lookup)
  ELSE(APPLE)
      SET (PYTHON_LINK_MODULE -shared)
  ENDIF(APPLE)

  include_directories(${PYTHON_NUMPY_INCLUDE_DIR})
  MESSAGE (STATUS "Numpy include in ${PYTHON_NUMPY_INCLUDE_DIR}")
  INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_DIRS})

ENDIF (PYTHON_FOUND)
