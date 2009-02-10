DEFAULT_INCLUDES = -I$(top_builddir)/src -I$(top_srcdir)/src
AM_CPPFLAGS = $(BASE_CPPFLAGS) $(BOOST_CPPFLAGS) $(XML_CPPFLAGS) $(HDF5_CPPFLAGS) $(MPI_CPPFLAGS)
AM_LDFLAGS = $(BASE_LDFLAGS) $(BOOST_LDFLAGS) $(HDF5_LDFLAGS) $(XML_LDFLAGS)
LDADD = $(top_builddir)/lib/alps/libalps.la $(top_builddir)/lib/comm-sgl/libcomm-sgl.la $(BOOST_LIBS) $(HDF5_LIBS) $(XML_LIBS) $(BASE_LIBS)

check_PROGRAMS =
if HAVE_HDF5
check_PROGRAMS += io
endif
io_SOURCES = io.C

dist_noinst_DATA =

.PHONY : test tests example examples
test tests : check
example examples :
