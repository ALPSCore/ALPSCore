DEFAULT_INCLUDES = -I$(top_builddir)/src -I$(top_srcdir)/src
AM_CPPFLAGS = $(BASE_CPPFLAGS) $(BOOST_CPPFLAGS) $(XML_CPPFLAGS) $(HDF5_CPPFLAGS)
AM_LDFLAGS = $(BASE_LDFLAGS) $(BOOST_LDFLAGS) $(HDF5_LDFLAGS) $(XML_LDFLAGS)
LDADD = $(top_builddir)/lib/alps/libalps.la $(top_builddir)/lib/comm-sgl/libcomm-sgl.la $(BOOST_LIBS) $(HDF5_LIBS) $(XML_LIBS) $(BASE_LIBS)

check_PROGRAMS = detailedbinning signed simpleobseval testobservableset vectorobseval

detailedbinning_SOURCES = detailedbinning.C
signed_SOURCES = signed.C
simpleobseval_SOURCES = simpleobseval.C
testobservableset_SOURCES = testobservableset.C
vectorobseval_SOURCES = vectorobseval.C

dist_noinst_DATA = detailedbinning.input detailedbinning.output signed.input signed.output simpleobseval.output testobservableset.input testobservableset.output vectorobseval.output

.PHONY : test tests example examples
test tests : check
example examples :
