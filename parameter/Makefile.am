DEFAULT_INCLUDES = -I$(top_builddir)/src -I$(top_srcdir)/src
AM_CPPFLAGS = $(BASE_CPPFLAGS) $(BOOST_CPPFLAGS) $(XML_CPPFLAGS) $(HDF5_CPPFLAGS)
AM_LDFLAGS = $(BASE_LDFLAGS) $(BOOST_LDFLAGS) $(HDF5_LDFLAGS) $(XML_LDFLAGS)
LDADD = $(top_builddir)/lib/alps/libalps.la $(top_builddir)/lib/comm-sgl/libcomm-sgl.la $(BOOST_LIBS) $(HDF5_LIBS) $(XML_LIBS) $(BASE_LIBS)

TESTS = expression expression2 flatten parameterlist parameterlist_xml parameters
TESTS_norun =
check_PROGRAMS = $(TESTS) $(TESTS_norun)
TESTS_ENVIRONMENT = $(top_srcdir)/config/run-test $(srcdir)

expression_SOURCES = expression.C
expression2_SOURCES = expression2.C
flatten_SOURCES = flatten.C
parameterlist_SOURCES = parameterlist.C
parameterlist_xml_SOURCES = parameterlist_xml.C
parameters_SOURCES = parameters.C

dist_noinst_DATA = expression.input expression.output expression2.output flatten.input flatten.output parameterlist.input parameterlist.output parameterlist_xml.input parameterlist_xml.output parameters.input parameters.output

.PHONY : test tests example examples
test tests : check
example examples :

clean-local :
	rm -f parameterlist.dump parameters.dump
