DEFAULT_INCLUDES = -I$(top_builddir)/src -I$(top_srcdir)/src
AM_CPPFLAGS = $(BASE_CPPFLAGS) $(BOOST_CPPFLAGS) $(XML_CPPFLAGS) $(HDF5_CPPFLAGS) $(MPI_CPPFLAGS)
AM_LDFLAGS = $(BASE_LDFLAGS) $(BOOST_LDFLAGS) $(SQLITE_LDFLAGS) $(XML_LDFLAGS) $(HDF5_LDFLAGS) $(MPI_LDFLAGS)
if BUILD_UNIFIED_LIB
  LDADD = $(top_builddir)/lib/alps/libalps-sgl.la $(BOOST_LIBS) $(SQLITE_LIBS) $(XML_LIBS) $(HDF5_LIBS) $(MPI_LIBS) $(BASE_LIBS)
else
  LDADD = $(top_builddir)/lib/alps/libalps.la $(BOOST_LIBS) $(SQLITE_LIBS) $(XML_LIBS) $(HDF5_LIBS) $(MPI_LIBS) $(BASE_LIBS) 
endif


check_PROGRAMS =
if HAVE_HDF5
check_PROGRAMS += io
check_PROGRAMS += alea_mocasito
endif
io_SOURCES = io.C
alea_mocasito_SOURCES = alea_mocasito.C

dist_noinst_DATA = alea_mocasito.output

.PHONY : test tests example examples
test tests : check
example examples :
