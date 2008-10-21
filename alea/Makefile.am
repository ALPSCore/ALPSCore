DEFAULT_INCLUDES = -I$(top_builddir)/src -I$(top_srcdir)/src
AM_CPPFLAGS = $(BASE_CPPFLAGS) $(BOOST_CPPFLAGS) $(XML_CPPFLAGS) $(HDF5_CPPFLAGS) $(MPI_CPPFLAGS)
AM_LDFLAGS = $(BASE_LDFLAGS) $(BOOST_LDFLAGS) $(HDF5_LDFLAGS) $(XML_LDFLAGS)
LDADD = $(top_builddir)/lib/alps/libalps.la $(top_builddir)/lib/comm-sgl/libcomm-sgl.la $(BOOST_LIBS) $(HDF5_LIBS) $(XML_LIBS) $(BASE_LIBS)

LDFLAGS_MPI = $(LDFLAGS_SGL) $(MPI_LDFLAGS)
LIBS_MPI = $(top_builddir)/lib/alps/libalps.la $(top_builddir)/lib/comm-mpi/libcomm-mpi.la $(BOOST_LIBS) $(HDF5_LIBS) $(XML_LIBS) $(MPI_LIBS) $(BASE_LIBS)

check_PROGRAMS = detailedbinning histogram signed simpleobseval testobservableset observableset_xml vectorobseval
if HAVE_BOOST_MPI
# check_PROGRAMS += observableset_mpi
endif
detailedbinning_SOURCES = detailedbinning.C
histogram_SOURCES = histogram.C
observableset_xml_SOURCES = observableset_xml.C
signed_SOURCES = signed.C
simpleobseval_SOURCES = simpleobseval.C
testobservableset_SOURCES = testobservableset.C
vectorobseval_SOURCES = vectorobseval.C

# observableset_mpi_SOURCES = observableset_mpi.C
# observableset_mpi_LDFLAGS = $(LDFLAGS_MPI)
# observableset_mpi_LDADD = $(LIBS_MPI)

dist_noinst_DATA = detailedbinning.input detailedbinning.output histogram.output observableset_xml.input observableset_xml.output signed.input signed.output simpleobseval.output testobservableset.input testobservableset.output vectorobseval.output
dist_noinst_DATA += observableset_mpi.op-2

.PHONY : test tests example examples
test tests : check
example examples :
