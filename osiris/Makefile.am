DEFAULT_INCLUDES = -I$(top_builddir)/src -I$(top_srcdir)/src
AM_CPPFLAGS = $(BASE_CPPFLAGS) $(BOOST_CPPFLAGS) $(XML_CPPFLAGS) $(HDF5_CPPFLAGS)
AM_LDFLAGS = $(BASE_LDFLAGS) $(BOOST_LDFLAGS) $(HDF5_LDFLAGS) $(XML_LDFLAGS)
LDADD = $(top_builddir)/lib/alps/libalps.la $(top_builddir)/lib/comm-sgl/libcomm-sgl.la $(BOOST_LIBS) $(HDF5_LIBS) $(XML_LIBS) $(BASE_LIBS)

TESTS = sizeof xdrdump xdrdump2
TESTS_norun =
check_PROGRAMS = $(TESTS) $(TESTS_norun)
TESTS_ENVIRONMENT = $(top_srcdir)/config/run-test $(srcdir)

sizeof_SOURCES = sizeof.C
xdrdump_SOURCES = xdrdump.C
xdrdump2_SOURCES = xdrdump2.C

dist_noinst_DATA = xdrdump.output xdrdump2.dump xdrdump2.output

.PHONY : test tests example examples
test tests : check
example examples :

clean-local :
	rm -f xdrdump.dump
