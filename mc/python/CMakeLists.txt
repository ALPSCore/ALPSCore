# 
# alps-mc cmake configuration script.
# 

cmake_minimum_required (VERSION 2.8)

# define project name

if (NOT DocumentationOnly)

  if (ENABLE_PYTHON)

    find_package(Python)

    if (PYTHON_FOUND)

      set(ALPS_SHARED_CPPFLAGS PYALPS_EXPORTS=1)  

      set(PYALPS_SOURCES pyrandom01_c)

      set_property(GLOBAL APPEND PROPERTY PY_MODULES_LIST pyrandom01_c)
      add_library(pyrandom01_c MODULE src/pyrandom01.cpp)

      set_target_properties(pyrandom01_c PROPERTIES COMPILE_DEFINITIONS "${ALPS_SHARED_CPPFLAGS}")
      set_target_properties(pyrandom01_c PROPERTIES PREFIX "")
      target_link_libraries(pyrandom01_c ${PYTHON_LIBRARY} ${LINK_ALL})

    else (PYTHON_FOUND)
      message(FATAL_ERROR " Python libraries not found")
    endif (PYTHON_FOUND)

  endif (ENABLE_PYTHON)

endif (NOT DocumentationOnly)
gen_documentation()
