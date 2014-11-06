# 
# alps-mc cmake configuration script.
# 

cmake_minimum_required (VERSION 2.8)

# define project name

if (NOT DocumentationOnly)

  if (ENABLE_PYTHON)

    find_package(PythonLibs)

    if (PYTHONLIBS_FOUND)

      set(ALPS_HAVE_PYTHON TRUE) # better put it through configure_file?
      set(ALPS_SHARED_CPPFLAGS PYALPS_EXPORTS=1)  


      foreach(name pyrandom01)
#      foreach(name pyrandom01 pymcbase)
        add_library(${name}_c MODULE src/${name}.cpp)

        set_target_properties(${name}_c PROPERTIES COMPILE_DEFINITIONS "${ALPS_SHARED_CPPFLAGS}")
        set_target_properties(${name}_c PROPERTIES PREFIX "")
        target_link_libraries(${name}_c ${PYTHON_LIBRARIES} ${LINK_ALL})
        include_directories(${PYTHON_INCLUDE_DIRS})
      endforeach(name)

    else (PYTHONLIBS_FOUND)
      message(FATAL_ERROR " Python libraries not found")
    endif (PYTHONLIBS_FOUND)

  endif (ENABLE_PYTHON)

endif (NOT DocumentationOnly)
gen_documentation()
