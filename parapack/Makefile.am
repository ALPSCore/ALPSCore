DEFAULT_INCLUDES = -I$(top_builddir)/src -I$(top_srcdir)/src $(BASE_CPPFLAGS) $(BOOST_CPPFLAGS) $(XML_CPPFLAGS) $(HDF5_CPPFLAGS) $(MPI_CPPFLAGS)

LDFLAGS_SGL = $(BASE_LDFLAGS) $(BOOST_LDFLAGS) $(HDF5_LDFLAGS) $(XML_LDFLAGS)
if BUILD_UNIFIED_LIB
  LIBS_SGL = $(top_builddir)/lib/alps/libalps-sgl.la $(BOOST_LIBS) $(HDF5_LIBS) $(XML_LIBS) $(BASE_LIBS)
else
  LIBS_SGL = $(top_builddir)/lib/alps/libparapack-sgl.la $(top_builddir)/lib/alps/libalps.la $(top_builddir)/lib/comm-sgl/libcomm-sgl.la $(BOOST_LIBS) $(HDF5_LIBS) $(XML_LIBS) $(BASE_LIBS)
endif

LDFLAGS_MPI = $(LDFLAGS_SGL) $(MPI_LDFLAGS)
if BUILD_UNIFIED_LIB
  LIBS_MPI = $(top_builddir)/lib/alps/libalps-mpi.la $(BOOST_LIBS) $(HDF5_LIBS) $(XML_LIBS) $(MPI_LIBS) $(BASE_LIBS)
else
  LIBS_MPI = $(top_builddir)/lib/alps/libparapack-mpi.la $(top_builddir)/lib/alps/libalps.la $(top_builddir)/lib/comm-mpi/libcomm-mpi.la $(BOOST_LIBS) $(HDF5_LIBS) $(XML_LIBS) $(MPI_LIBS) $(BASE_LIBS)
endif

dist_noinst_DATA = \
	clone_mpi.ip-4 \
	clone_info.ip clone_info.op \
	clone_phase.ip clone_phase.op \
	collect_mpi.ip-4 \
	exmc_optimize.op \
	exp_number.op \
	filelock_mpi.ip-4 \
	id2string.op \
	integer_range.ip integer_range.op \
	linear_regression.op \
	merge.op \
	percentage.op \
	process_mpi.op-8 \
	temperature_scan.ip temperature_scan.op \
	wl_weight.op \
	worker_mpi.ip-4 worker_mpi.op-4

check_PROGRAMS = \
	clone_info \
	clone_phase \
	clone_timer \
	exmc_optimize \
	exp_number \
	id2string \
	info_test \
	integer_range \
	linear_regression \
	merge \
	percentage \
	temperature_scan \
	time \
	version \
	wl_weight

if HAVE_BOOST_MPI
check_PROGRAMS += \
	comm_mpi \
	collect_mpi \
	filelock_mpi \
	halt_mpi \
	info_test_mpi \
	process_mpi
#	clone_mpi
#	worker_mpi
endif

# clone_mpi_SOURCES = clone_mpi.C
# clone_mpi_LDFLAGS = $(LDFLAGS_MPI)
# clone_mpi_LDADD = $(LIBS_MPI)

clone_info_SOURCES = clone_info.C
clone_info_LDFLAGS = $(LDFLAGS_SGL)
clone_info_LDADD = $(LIBS_SGL)

clone_phase_SOURCES = clone_phase.C
clone_phase_LDFLAGS = $(LDFLAGS_SGL)
clone_phase_LDADD = $(LIBS_SGL)

clone_timer_SOURCES = clone_timer.C
clone_timer_LDFLAGS = $(LDFLAGS_SGL)
clone_timer_LDADD = $(LIBS_SGL)

comm_mpi_SOURCES = comm_mpi.C
comm_mpi_LDFLAGS = $(LDFLAGS_MPI)
comm_mpi_LDADD = $(LIBS_MPI)

collect_mpi_SOURCES = collect_mpi.C
collect_mpi_LDFLAGS = $(LDFLAGS_MPI)
collect_mpi_LDADD = $(LIBS_MPI)

exmc_optimize_SOURCES = exmc_optimize.C
exmc_optimize_LDFLAGS = $(LDFLAGS_SGL)
exmc_optimize_LDADD = $(LIBS_SGL)

exp_number_SOURCES = exp_number.C
exp_number_LDFLAGS = $(LDFLAGS_SGL)
exp_number_LDADD = $(LIBS_SGL)

filelock_mpi_SOURCES = filelock_mpi.C
filelock_mpi_LDFLAGS = $(LDFLAGS_MPI)
filelock_mpi_LDADD = $(LIBS_MPI)

id2string_SOURCES = id2string.C
id2string_LDFLAGS = $(LDFLAGS_SGL)
id2string_LDADD = $(LIBS_SGL)

info_test_SOURCES = info_test.C
info_test_LDFLAGS = $(LDFLAGS_SGL)
info_test_LDADD = $(LIBS_SGL)

info_test_mpi_SOURCES = info_test_mpi.C
info_test_mpi_LDFLAGS = $(LDFLAGS_MPI)
info_test_mpi_LDADD = $(LIBS_MPI)

integer_range_SOURCES = integer_range.C
integer_range_LDFLAGS = $(LDFLAGS_SGL)
integer_range_LDADD = $(LIBS_SGL)

linear_regression_SOURCES = linear_regression.C
linear_regression_LDFLAGS = $(LDFLAGS_SGL)
linear_regression_LDADD = $(LIBS_SGL)

merge_SOURCES = merge.C
merge_LDFLAGS = $(LDFLAGS_SGL)
merge_LDADD = $(LIBS_SGL)

halt_mpi_SOURCES = halt_mpi.C
halt_mpi_LDFLAGS = $(LDFLAGS_MPI)
halt_mpi_LDADD = $(LIBS_MPI)

percentage_SOURCES = percentage.C
percentage_LDFLAGS = $(LDFLAGS_SGL)
percentage_LDADD = $(LIBS_SGL)

process_mpi_SOURCES = process_mpi.C
process_mpi_LDFLAGS = $(LDFLAGS_MPI)
process_mpi_LDADD = $(LIBS_MPI)

temperature_scan_SOURCES = temperature_scan.C
temperature_scan_LDFLAGS = $(LDFLAGS_SGL)
temperature_scan_LDADD = $(LIBS_SGL)

time_SOURCES = time.C
time_LDFLAGS = $(LDFLAGS_SGL)
time_LDADD = $(LIBS_SGL)

version_SOURCES = version.C
version_LDFLAGS = $(LDFLAGS_SGL)
version_LDADD = $(LIBS_SGL)

wl_weight_SOURCES = wl_weight.C
wl_weight_LDFLAGS = $(LDFLAGS_SGL)
wl_weight_LDADD = $(LIBS_SGL)

# worker_mpi_SOURCES = worker_mpi.C
# worker_mpi_LDFLAGS = $(LDFLAGS_MPI)
# worker_mpi_LDADD = $(LIBS_MPI)

.PHONY : test tests example examples
test tests : check
example examples :

clean-local :
	rm -rf wl_weight.dump0 wl_weight.dump1
