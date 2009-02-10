DEFAULT_INCLUDES = -I$(top_builddir)/src -I$(top_srcdir)/src
AM_CPPFLAGS = $(BASE_CPPFLAGS) $(BOOST_CPPFLAGS) $(XML_CPPFLAGS) $(HDF5_CPPFLAGS) $(MPI_CPPFLAGS)
AM_LDFLAGS = $(BASE_LDFLAGS) $(BOOST_LDFLAGS) $(HDF5_LDFLAGS) $(XML_LDFLAGS)
LDADD = $(top_builddir)/lib/alps/libparapack-sgl.la $(top_builddir)/lib/alps/libalps.la $(top_builddir)/lib/comm-sgl/libcomm-sgl.la $(BOOST_LIBS) $(HDF5_LIBS) $(XML_LIBS) $(BASE_LIBS)

LDFLAGS_MPI = $(BASE_LDFLAGS) $(BOOST_LDFLAGS) $(HDF5_LDFLAGS) $(XML_LDFLAGS) $(MPI_LDFLAGS)
LIBS_MPI = $(top_builddir)/lib/alps/libparapack-mpi.la $(top_builddir)/lib/alps/libalps.la $(top_builddir)/lib/comm-mpi/libcomm-mpi.la $(BOOST_LIBS) $(HDF5_LIBS) $(XML_LIBS) $(MPI_LIBS) $(BASE_LIBS)

check_PROGRAMS = expression expression2 flatten parameter parameterlist parameterlist_xml parameters parameters_xml
if HAVE_BOOST_MPI
check_PROGRAMS += parameters_mpi
endif
expression_SOURCES = expression.C
expression2_SOURCES = expression2.C
flatten_SOURCES = flatten.C
parameter_SOURCES = parameter.C
parameterlist_SOURCES = parameterlist.C
parameterlist_xml_SOURCES = parameterlist_xml.C
parameters_SOURCES = parameters.C
parameters_xml_SOURCES = parameters_xml.C

parameters_mpi_SOURCES = parameters_mpi.C
parameters_mpi_LDFLAGS = $(LDFLAGS_MPI)
parameters_mpi_LDADD = $(LIBS_MPI)

dist_noinst_DATA = expression.input expression.output expression2.output flatten.input flatten.output parameter.input parameter.output parameterlist.input parameterlist.output parameterlist_xml.input parameterlist_xml.output parameters.input parameters.output parameters_xml.input parameters_xml.output
dist_noinst_DATA += parameters_mpi.ip-2 parameters_mpi.op-2

.PHONY : test tests example examples
test tests : check
example examples :

clean-local :
	rm -f parameterlist.dump parameters.dump
