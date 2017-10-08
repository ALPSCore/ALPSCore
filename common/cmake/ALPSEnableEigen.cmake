# enable using Eigen
option(ALPS_USE_EIGEN "Use Eigen3 library bundled with ALPSCore" ON)
mark_as_advanced(ALPS_USE_EIGEN)

# Add eigen to the current module (target ${PROJECT_NAME})
function(add_eigen)
  message("DEBUG: eigen requested")

  # nested function to determine Eigen version
  function(get_eigen_version_ eigen_dir return_var_name)
    message("DEBUG: determining Eigen version...")
    # the code is borrowed from Eigen's CMakeLists.txt
    file(READ "${eigen_dir}/Eigen/src/Core/util/Macros.h" _eigen_version_header)
    string(REGEX MATCH "define[ \t]+EIGEN_WORLD_VERSION[ \t]+([0-9]+)" _eigen_world_version_match "${_eigen_version_header}")
    set(EIGEN_WORLD_VERSION "${CMAKE_MATCH_1}")
    string(REGEX MATCH "define[ \t]+EIGEN_MAJOR_VERSION[ \t]+([0-9]+)" _eigen_major_version_match "${_eigen_version_header}")
    set(EIGEN_MAJOR_VERSION "${CMAKE_MATCH_1}")
    string(REGEX MATCH "define[ \t]+EIGEN_MINOR_VERSION[ \t]+([0-9]+)" _eigen_minor_version_match "${_eigen_version_header}")
    set(EIGEN_MINOR_VERSION "${CMAKE_MATCH_1}")
    set(EIGEN_VERSION_NUMBER ${EIGEN_WORLD_VERSION}.${EIGEN_MAJOR_VERSION}.${EIGEN_MINOR_VERSION})
    message("DEBUG: ...it's ${EIGEN_VERSION_NUMBER}")
    set(${return_var_name} ${EIGEN_VERSION_NUMBER} PARENT_SCOPE)
  endfunction(get_eigen_version_)
  
  # if (NOT ALPS_HAVE_EIGEN_VERSION)
  #   # we have not yet found EIGEN, try to do it
  #   message("DEBUG: trying to locate Eigen")
  if (ALPS_USE_EIGEN)
    # use the bundled version
    set(eigen_dir "${CMAKE_SOURCE_DIR}/common/deps/eigen-eigen-5a0156e40feb")
    message("DEBUG: using the bundled version in ${eigen_dir}")
    if (NOT ALPS_HAVE_EIGEN_VERSION)
      get_eigen_version_(${eigen_dir} ALPS_HAVE_EIGEN_VERSION)
      set(ALPS_HAVE_EIGEN_VERSION ${ALPS_HAVE_EIGEN_VERSION} CACHE INTERNAL "The Eigen version used by ALPSCore")
    endif()
    message("DEBUG: the bundled version is ${ALPS_HAVE_EIGEN_VERSION}")

    if (NOT TARGET eigen)
      message("DEBUG: setting the `eigen` target to bundled Eigen")
      # Create the interface target and set up installation
      add_library(eigen INTERFACE)

      set(eigen_install_dir "${CMAKE_INSTALL_PREFIX}/alps/deps/eigen")
      
      target_include_directories(eigen INTERFACE
        $<BUILD_INTERFACE:${eigen_dir}>
        $<INSTALL_INTERFACE:${eigen_install_dir}>)
      
      install(TARGETS eigen EXPORT eigen INCLUDES DESTINATION ".")
      install(EXPORT eigen DESTINATION "share/ALPSCore" NAMESPACE alps::)
      install(DIRECTORY "${eigen_dir}/Eigen" "${eigen_dir}/unsupported" DESTINATION ${eigen_install_dir})
    endif()
    target_link_libraries(${PROJECT_NAME} PUBLIC eigen)

  else(ALPS_USE_EIGEN)

    message("DEBUG: an external Eigen requested; EIGEN_INCLUDE_DIR=${EIGEN_INCLUDE_DIR} ENV{EIGEN_INCLUDE_DIR}=$ENV{EIGEN_INCLUDE_DIR}")

    set(env_ $ENV{EIGEN_INCLUDE_DIR})
    if (NOT EIGEN_INCLUDE_DIR AND env_)
      set(EIGEN_INCLUDE_DIR $ENV{EIGEN_INCLUDE_DIR})
      message("DEBUG: the Eigen location is set from the environment")
    endif()
    
    if (EIGEN_INCLUDE_DIR)
      message("DEBUG: external Eigen is in ${EIGEN_INCLUDE_DIR}")
      if (NOT ALPS_HAVE_EIGEN_VERSION)
        get_eigen_version_(${EIGEN_INCLUDE_DIR} ALPS_HAVE_EIGEN_VERSION)
        if (NOT ALPS_HAVE_EIGEN_VERSION)
          message(FATAL_ERROR "Cannot find Eigen at ${EIGEN_INCLUDE_DIR}")
        endif()
      endif()
      message("DEBUG: the external version is ${ALPS_HAVE_EIGEN_VERSION}")
      set(ALPS_HAVE_EIGEN_VERSION ${ALPS_HAVE_EIGEN_VERSION} CACHE INTERNAL "The Eigen version used by ALPSCore")
        
      # Create the imported target
      if (NOT TARGET eigen)
        message("DEBUG: setting the `eigen` target to external Eigen")
        add_library(eigen INTERFACE IMPORTED GLOBAL)
        set(dependency_on_eigen "eigen")
        
        set_target_properties(eigen PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${EIGEN_INCLUDE_DIR})
      endif()
      target_link_libraries(${PROJECT_NAME} PUBLIC eigen)
        
    else(EIGEN_INCLUDE_DIR)

      message("DEBUG: trying to locate external Eigen3")
      find_package(Eigen3 REQUIRED)
      set(ALPS_HAVE_EIGEN_VERSION ${Eigen3_VERSION})
      message("DEBUG: found external Eigen3, version is ${ALPS_HAVE_EIGEN_VERSION}")

      # The imported target should be available
      if (NOT TARGET Eigen3::Eigen)
        message(FATAL_ERROR "The expected target `Eigen3::Eigen` is not defined by the Eigen3 package, "
          "try to use bundled-in Eigen3 version and/or report this problem to ALPSCore developers")
      endif()
      target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen)
      set(ALPS_HAVE_EIGEN_VERSION ${ALPS_HAVE_EIGEN_VERSION} CACHE INTERNAL "The Eigen version used by ALPSCore")
      
      # if (NOT TARGET eigen)
      #   message("DEBUG: setting the `eigen` target to imported Eigen3::Eigen")
      #   add_library(eigen INTERFACE IMPORTED GLOBAL)
      #   target_link_libraries(eigen  Eigen3::Eigen)
      #   # install(TARGETS eigen EXPORT ${PROJECT_NAME})
      # endif()
        
    endif(EIGEN_INCLUDE_DIR)

  endif(ALPS_USE_EIGEN)

endfunction()
