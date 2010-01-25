#
# libalps
#

set(ALPS_SOURCES "")
set(ALPS_SHARED_CPPFLAGS ALPS_EXPORTS=1)
set(ALPS_STATIC_CPPFLAGS "")

if (ALPS_FOR_VISTRAILS)
set(ALPS_SOURCES
    # alps 
    copyright.C encode.cpp
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
    osiris/buffer.C osiris/dump.C osiris/dumparchive.C osiris/os.C osiris/process.C osiris/comm.C
    osiris/mpdump.C osiris/xdrdump.C osiris/xdrcore.C
    # parameter
    parameter/parameter.C parameter/parameterlist.C parameter/parameters.C
    # parapack
    parapack/clone.C parapack/clone_info.C parapack/filelock.C parapack/logger.C
    parapack/job.C parapack/mc_worker.C parapack/measurement.C parapack/process_impl.C
    parapack/option.C parapack/parapack.C parapack/queue.C parapack/rng_helper.C
    parapack/types.C parapack/util.C parapack/worker_factory.C parapack/version.C
    # parser
    parser/parser.C parser/xmlattributes.C parser/xmlhandler.C parser/xmlparser.C
    parser/xmlstream.C parser/xslt_path.C
    # random
    random/rngfactory.C
    # scheduler
    scheduler/abstract_task.C scheduler/convertxdr.C scheduler/factory.C scheduler/info.C
    scheduler/master_scheduler.C scheduler/measurement_operators.C
    scheduler/montecarlo.C scheduler/mpp_scheduler.C scheduler/options.C
    scheduler/remote_task.C scheduler/remote_worker.C scheduler/scheduler.C
    scheduler/serial_scheduler.C scheduler/signal.C scheduler/single_scheduler.C
    scheduler/slave_task.C scheduler/task.C scheduler/worker.C scheduler/workertask.C
)
else(ALPS_FOR_VISTRAILS)
# alps
set(ALPS_SOURCES
    # alps 
    copyright.C encode.cpp
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
    osiris/buffer.C osiris/dump.C osiris/dumparchive.C osiris/os.C osiris/process.C osiris/comm.C
    osiris/mpdump.C osiris/xdrdump.C osiris/xdrcore.C
    # parameter
    parameter/parameter.C parameter/parameterlist.C parameter/parameters.C
    # parapack
    parapack/clone.C parapack/clone_info.C parapack/filelock.C parapack/logger.C
    parapack/job.C parapack/mc_worker.C parapack/measurement.C parapack/process_impl.C
    parapack/option.C parapack/parapack.C parapack/queue.C parapack/rng_helper.C
    parapack/types.C parapack/util.C parapack/version.C parapack/worker_factory.C
    # parser
    parser/parser.C parser/xmlattributes.C parser/xmlhandler.C parser/xmlparser.C
    parser/xmlstream.C parser/xslt_path.C
    # random
    random/rngfactory.C
    # scheduler
    scheduler/abstract_task.C scheduler/convertxdr.C scheduler/factory.C scheduler/info.C
    scheduler/master_scheduler.C scheduler/measurement_operators.C
    scheduler/montecarlo.C scheduler/mpp_scheduler.C scheduler/options.C
    scheduler/remote_task.C scheduler/remote_worker.C scheduler/scheduler.C
    scheduler/serial_scheduler.C scheduler/signal.C scheduler/single_scheduler.C
    scheduler/slave_task.C scheduler/task.C scheduler/worker.C scheduler/workertask.C
)
endif(ALPS_FOR_VISTRAILS)

add_library(alps ${ALPS_SOURCES})

target_link_libraries(alps boost ${LINK_LIBRARIES})
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

if (ALPS_FOR_VISTRAILS)
  SET_TARGET_PROPERTIES(alps
    PROPERTIES BUILD_WITH_INSTALL_RPATH ON
    INSTALL_NAME_DIR "@executable_path/../Frameworks"
  )
endif (ALPS_FOR_VISTRAILS)


#######################################################################
# install
#######################################################################
if (ALPS_FOR_VISTRAILS)
  install(TARGETS alps
          ARCHIVE DESTINATION ${VISTRAILS_APP_NAME}/${VISTRAILS_DYLIB_DIR}
          LIBRARY DESTINATION ${VISTRAILS_APP_NAME}/${VISTRAILS_DYLIB_DIR}
          RUNTIME DESTINATION ${VISTRAILS_APP_NAME}/${VISTRAILS_DYLIB_DIR})
else (ALPS_FOR_VISTRAILS)
  install(TARGETS alps COMPONENT libraries
          ARCHIVE DESTINATION lib
          LIBRARY DESTINATION lib
          RUNTIME DESTINATION bin)
endif (ALPS_FOR_VISTRAILS)
