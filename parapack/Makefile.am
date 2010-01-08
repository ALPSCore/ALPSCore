DEFAULT_INCLUDES = -I$(top_builddir)/src -I$(top_srcdir)/src
AM_CPPFLAGS = $(BASE_CPPFLAGS) $(BOOST_CPPFLAGS) $(XML_CPPFLAGS) $(HDF5_CPPFLAGS) $(MPI_CPPFLAGS)
AM_LDFLAGS = $(BASE_LDFLAGS) $(BOOST_LDFLAGS) $(SQLITE_LDFLAGS) $(XML_LDFLAGS) $(HDF5_LDFLAGS) $(MPI_LDFLAGS)
LDADD = $(top_builddir)/lib/alps/libalps.la $(BOOST_LIBS) $(SQLITE_LIBS) $(XML_LIBS) $(HDF5_LIBS) $(MPI_LIBS) $(BASE_LIBS) 

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
	footprint \
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
clone_info_SOURCES = clone_info.C
clone_phase_SOURCES = clone_phase.C
clone_timer_SOURCES = clone_timer.C
comm_mpi_SOURCES = comm_mpi.C
collect_mpi_SOURCES = collect_mpi.C
exmc_optimize_SOURCES = exmc_optimize.C
exp_number_SOURCES = exp_number.C
filelock_mpi_SOURCES = filelock_mpi.C
footprint_SOURCES = footprint.C
id2string_SOURCES = id2string.C
info_test_SOURCES = info_test.C
info_test_mpi_SOURCES = info_test_mpi.C
integer_range_SOURCES = integer_range.C
linear_regression_SOURCES = linear_regression.C
merge_SOURCES = merge.C
halt_mpi_SOURCES = halt_mpi.C
percentage_SOURCES = percentage.C
process_mpi_SOURCES = process_mpi.C
temperature_scan_SOURCES = temperature_scan.C
time_SOURCES = time.C
version_SOURCES = version.C
wl_weight_SOURCES = wl_weight.C
# worker_mpi_SOURCES = worker_mpi.C

.PHONY : test tests example examples
test tests : check
example examples :

clean-local :
	rm -rf wl_weight.dump0 wl_weight.dump1
