DEFAULT_INCLUDES = -I$(top_builddir)/src -I$(top_srcdir)/src
AM_CPPFLAGS = $(BASE_CPPFLAGS) $(BOOST_CPPFLAGS) $(XML_CPPFLAGS) $(HDF5_CPPFLAGS)
AM_LDFLAGS = $(BASE_LDFLAGS) $(BOOST_LDFLAGS) $(HDF5_LDFLAGS) $(XML_LDFLAGS)
LDADD = $(top_builddir)/lib/alps/libalps.la $(top_builddir)/lib/comm-sgl/libcomm-sgl.la $(BOOST_LIBS) $(HDF5_LIBS) $(XML_LIBS) $(BASE_LIBS)

TESTS = xmlhandler xmlparser xmlstream
TESTS_norun =
check_PROGRAMS = $(TESTS) $(TESTS_norun)
TESTS_ENVIRONMENT = $(top_srcdir)/config/run-test $(srcdir)

xmlhandler_SOURCES = xmlhandler.C
xmlparser_SOURCES = xmlparser.C
xmlstream_SOURCES = xmlstream.C

dist_noinst_DATA = xmlhandler.input xmlhandler.output xmlparser.input xmlparser.output xmlstream.output

.PHONY : test tests example examples
test tests : check
example examples :
