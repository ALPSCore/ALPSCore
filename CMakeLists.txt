#  Copyright Matthias Troyer 2009 - 2010.
#  Distributed under the Boost Software License, Version 1.0.
#      (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

#
# libalps
#

set(ALPS_SOURCES "")
set(ALPS_SHARED_CPPFLAGS ALPS_EXPORTS=1)
set(ALPS_STATIC_CPPFLAGS "")

set(ALPS_SOURCES
    # utility
    utility/encode.cpp utility/copyright.cpp utility/os.cpp utility/temporary_filename.cpp
    # alea
    alea/nan.C alea/observable.C alea/observablefactory.C alea/observableset.C 
    # expression
    expression/evaluator.C
    # lattice
    lattice/disorder.C lattice/latticedescriptor.C lattice/latticegraphdescriptor.C
    lattice/latticelibrary.C lattice/unitcell.C
    # model
    model/bondterm.C model/globaloperator.C model/modellibrary.C model/siteterm.C
    # osiris
    osiris/buffer.C osiris/dump.C osiris/dumparchive.C osiris/process.C osiris/comm.C
    osiris/mpdump.C osiris/xdrdump.C osiris/xdrcore.C
    # parameter
    parameter/parameter.C parameter/parameterlist.C parameter/parameters.C
    # parser
    parser/parser.C parser/xmlattributes.C parser/xmlhandler.C parser/xmlparser.C
    parser/xmlstream.C parser/xslt_path.C
    #hdf5
    hdf5/hdf5.cpp
    # random
    random/rngfactory.C random/parallel/get_prime.cpp random/parallel/primelist_64.cpp
    # scheduler
    scheduler/abstract_task.C scheduler/convertxdr.C scheduler/factory.C scheduler/info.C
    scheduler/master_scheduler.C scheduler/measurement_operators.C
    scheduler/montecarlo.C scheduler/mpp_scheduler.C scheduler/options.C
    scheduler/remote_task.C scheduler/remote_worker.C scheduler/scheduler.C
    scheduler/serial_scheduler.C scheduler/signal.C scheduler/single_scheduler.C
    scheduler/slave_task.C scheduler/task.C scheduler/worker.C scheduler/workertask.C
)

if (ALPS_BUILD_NGS)
  set(ALPS_SOURCES ${ALPS_SOURCES} ngs/mcbase.cpp ngs/mcparams.cpp ngs/mcsignal.cpp ngs/mcoptions.cpp 
      ngs/mcdeprecated.cpp ngs/mcresult.cpp ngs/mcresults.cpp)
endif (ALPS_BUILD_NGS)

if (NOT ALPS_FOR_VISTRAILS)
  set(ALPS_SOURCES ${ALPS_SOURCES} 
      # parapack
      parapack/clone.C parapack/clone_info.C parapack/filelock.C parapack/logger.C
      parapack/job.C parapack/mc_worker.C parapack/measurement.C parapack/process_impl.C
      parapack/option.C parapack/parapack.C parapack/queue.C parapack/rng_helper.C
      parapack/types.C parapack/util.C parapack/version.C parapack/worker_factory.C
  )
endif(NOT ALPS_FOR_VISTRAILS)

add_library(alps ${ALPS_SOURCES})

find_package(HDF5)
if (Boost_FOUND)
  # link to ${Boost_LIBRARIES} when precompiled Boost libraries found
  set(ALPS_LINK_LIBS ${Boost_LIBRARIES} ${HDF5_LIBRARIES}) # ${SZIP_LIBRARIES} ${ZLIB_LIBRARIES})
  if(PYTHONLIBS_FOUND)
    set(ALPS_LINK_LIBS ${ALPS_LINK_LIBS} ${PYTHON_LIBRARY} ${PYTHON_EXTRA_LIBS})
  endif(PYTHONLIBS_FOUND)
  if(MPI_FOUND)
    set(ALPS_LINK_LIBS ${ALPS_LINK_LIBS} ${MPI_LIBRARIES} ${MPI_EXTRA_LIBRARY})
  endif(MPI_FOUND)
  target_link_libraries(alps ${ALPS_LINK_LIBS})
else (Boost_FOUND)
  # "boost" target available when Boost libraries are built from source
  target_link_libraries(alps boost ${HDF5_LIBRARIES}) # ${SZIP_LIBRARIES} ${ZLIB_LIBRARIES})
endif (Boost_FOUND)

if(BUILD_SHARED_LIBS)
  set_target_properties(alps PROPERTIES COMPILE_DEFINITIONS "${ALPS_SHARED_CPPFLAGS}")
else(BUILD_SHARED_LIBS)
  set_target_properties(alps PROPERTIES COMPILE_DEFINITIONS "${ALPS_STATIC_CPPFLAGS}")
endif(BUILD_SHARED_LIBS)

if(MSVC)
  get_target_property(LIB_NAME alps LOCATION)
  add_custom_command(TARGET alps POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${LIB_NAME} ${PROJECT_BINARY_DIR}/bin)
endif(MSVC)

# Set soversion for library
if(NOT WIN32 AND NOT APPLE)
  set_target_properties(alps
    PROPERTIES
    SOVERSION "${ALPS_VERSION_MAJOR}"
    VERSION "${ALPS_VERSION_MAJOR}.${ALPS_VERSION_MINOR}.${ALPS_VERSION_PATCH}" )
endif(NOT WIN32 AND NOT APPLE)

#######################################################################
# install
#######################################################################
if (NOT ALPS_FOR_VISTRAILS)
  install(TARGETS alps COMPONENT libraries
          ARCHIVE DESTINATION lib
          LIBRARY DESTINATION lib
          RUNTIME DESTINATION bin)
endif (NOT ALPS_FOR_VISTRAILS)

if (ALPS_INSTALL_VISTRAILS_PACKAGES)
  if (APPLE)
    SET_TARGET_PROPERTIES(alps
      PROPERTIES BUILD_WITH_INSTALL_RPATH ON
      INSTALL_NAME_DIR "@executable_path/../Frameworks"
    )
  endif(APPLE)
  install(TARGETS alps  COMPONENT vistrails
          ARCHIVE DESTINATION ${VISTRAILS_DYLIB_DIR}
          LIBRARY DESTINATION ${VISTRAILS_DYLIB_DIR}
          RUNTIME DESTINATION ${VISTRAILS_DYLIB_DIR})
endif (ALPS_INSTALL_VISTRAILS_PACKAGES)
