SUBDIRS = fixed_capacity osiris parameter parser alea mocasito
if ALPS_FULL
SUBDIRS += lattice model parapack
endif
SUBDIRS += .
DIST_SUBDIRS = $(SUBDIRS)

TESTS_ENVIRONMENT = $(top_srcdir)/config/run-test $(srcdir)

TESTS_FC = fixed_capacity/fixed_capacity_deque fixed_capacity/fixed_capacity_traits fixed_capacity/fixed_capacity_vector fixed_capacity/test_deque fixed_capacity/test_vector
TESTS_OSIRIS = osiris/os osiris/sizeof osiris/xdrdump osiris/xdrdump2 osiris/boostdump osiris/boostdump2 osiris/boostdump3 osiris/boostdump4
TESTS_PARAMETER = parameter/expression parameter/expression2 parameter/flatten parameter/parameter parameter/parameterlist parameter/parameterlist_xml parameter/parameters parameter/parameters_xml
if HAVE_BOOST_MPI
# TESTS_PARAMETER += parameter/parameters_mpi
endif
TESTS_PARSER = parser/xmlhandler parser/xmlparser parser/xmlstream
TESTS_ALEA = alea/detailedbinning alea/histogram alea/observableset_xml alea/signed alea/simpleobseval alea/testobservableset alea/vectorobseval
if HAVE_BOOST_MPI
# TESTS_ALEA += alea/observableset_mpi
endif
TESTS_MOCASITO =
if HAVE_HDF5
TESTS_MOCASITO += mocasito/io mocasito/alea_mocasito
endif
TESTS_LATTICE = lattice/example1 lattice/example2 lattice/example3 lattice/example4 lattice/example5 lattice/example6 lattice/example7 lattice/example8 lattice/example9 lattice/example10 lattice/label lattice/parity
TESTS_MODEL = model/example1 model/example2 model/example3 model/example4 model/example5 model/example6 model/example7 model/example8 model/example9 model/example10 model/example11 model/example12 model/example13 model/example14 model/example15 model/example16 model/example17 model/example18
TESTS_PARAPACK = parapack/alea parapack/clone_info parapack/clone_phase parapack/exmc_optimize parapack/exp_number parapack/id2string parapack/info_test parapack/integer_range parapack/linear_regression parapack/percentage parapack/time parapack/version parapack/wl_weight
if HAVE_BOOST_MPI
# TESTS_PARAPACK += parapack/clone_mpi parapack/comm_mpi parapack/collect_mpi parapack/filelock_mpi parapack/halt_mpi parapack/info_test_mpi parapack/process_mpi parapack/worker_mpi
endif

TESTS = $(TESTS_FC) $(TESTS_OSIRIS) $(TESTS_PARAMETER) $(TESTS_PARSER) $(TESTS_ALEA) $(TESTS_MOCASITO)
if ALPS_FULL
TESTS += $(TESTS_LATTICE) $(TESTS_MODEL) $(TESTS_PARAPACK)
endif

.PHONY : test tests example examples
test tests : check
example examples :
