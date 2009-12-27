DEFAULT_INCLUDES = -I$(top_builddir)/src -I$(top_srcdir)/src $(BASE_CPPFLAGS) $(BOOST_CPPFLAGS) $(XML_CPPFLAGS) $(SQLITE_CPPFLAGS) $(HDF5_CPPFLAGS) $(MPI_CPPFLAGS)

AM_LDFLAGS = $(BASE_LDFLAGS) $(BOOST_LDFLAGS) $(SQLITE_LDFLAGS) $(XML_LDFLAGS) $(HDF5_LDFLAGS) $(MPI_LDFLAGS)
if BUILD_UNIFIED_LIB
  LDADD = $(top_builddir)/lib/alps/libalps-sgl.la $(BOOST_LIBS) $(SQLITE_LIBS) $(XML_LIBS) $(HDF5_LIBS) $(MPI_LIBS) $(BASE_LIBS)
else
  LDADD = $(top_builddir)/lib/alps/libalps.la $(BOOST_LIBS) $(SQLITE_LIBS) $(XML_LIBS) $(HDF5_LIBS) $(MPI_LIBS) $(BASE_LIBS) 
endif

check_PROGRAMS = xmlhandler xmlparser xmlstream

xmlhandler_SOURCES = xmlhandler.C
xmlparser_SOURCES = xmlparser.C
xmlstream_SOURCES = xmlstream.C

dist_noinst_DATA = xmlhandler.input xmlhandler.output xmlparser.input xmlparser.output xmlstream.output

.PHONY : test tests example examples
test tests : check
example examples :
