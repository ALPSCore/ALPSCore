DEFAULT_INCLUDES = -I$(top_builddir)/src -I$(top_srcdir)/src
AM_CPPFLAGS = $(BASE_CPPFLAGS) $(BOOST_CPPFLAGS) $(XML_CPPFLAGS) $(HDF5_CPPFLAGS)
AM_LDFLAGS = $(BASE_LDFLAGS) $(BOOST_LDFLAGS) $(HDF5_LDFLAGS) $(XML_LDFLAGS)
LDADD = $(top_builddir)/lib/alps/libalps.la $(top_builddir)/lib/comm-sgl/libcomm-sgl.la $(BOOST_LIBS) $(HDF5_LIBS) $(XML_LIBS) $(BASE_LIBS)

TESTS = example1 example2 example3 example4 example5 example6 parity
TESTS_norun =
check_PROGRAMS = $(TESTS) $(TESTS_norun)
TESTS_ENVIRONMENT = $(top_srcdir)/config/run-test $(srcdir)

example1_SOURCES = example1.C
example2_SOURCES = example2.C
example3_SOURCES = example3.C
example4_SOURCES = example4.C
example5_SOURCES = example5.C
example6_SOURCES = example6.C
parity_SOURCES = parity.C

dist_noinst_DATA = example1.input example1.output example2.input example2.output example3.input example3.output example4.output example5.output example6.output parity.input parity.output

.PHONY : test tests example examples
test tests : check
example examples :
