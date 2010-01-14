DEFAULT_INCLUDES = -I$(top_builddir)/src -I$(top_srcdir)/src
AM_CPPFLAGS = $(BASE_CPPFLAGS) $(BOOST_CPPFLAGS) $(XML_CPPFLAGS) $(HDF5_CPPFLAGS) $(MPI_CPPFLAGS)
AM_LDFLAGS = $(BASE_LDFLAGS) $(BOOST_LDFLAGS) $(SQLITE_LDFLAGS) $(XML_LDFLAGS) $(HDF5_LDFLAGS) $(MPI_LDFLAGS)
LDADD = $(top_builddir)/lib/alps/libalps.la $(BOOST_LIBS) $(SQLITE_LIBS) $(XML_LIBS) $(HDF5_LIBS) $(MPI_LIBS) $(BASE_LIBS) 

check_PROGRAMS = expression expression2 flatten parameter parameterlist parameterlist_xml parameters parameters_xml
if HAVE_HDF5
check_PROGRAMS += parameters_hdf5
endif
# if HAVE_BOOST_MPI
# check_PROGRAMS += parameters_mpi
# endif

expression_SOURCES = expression.C
expression2_SOURCES = expression2.C
flatten_SOURCES = flatten.C
parameter_SOURCES = parameter.C
parameterlist_SOURCES = parameterlist.C
parameterlist_xml_SOURCES = parameterlist_xml.C
parameters_SOURCES = parameters.C
parameters_xml_SOURCES = parameters_xml.C
parameters_hdf5_SOURCES = parameters_hdf5.C

# parameters_mpi_SOURCES = parameters_mpi.C

dist_noinst_DATA = expression.input expression.output expression2.output flatten.input flatten.output parameter.input parameter.output parameterlist.input parameterlist.output parameterlist_xml.input parameterlist_xml.output parameters.input parameters.output parameters_xml.input parameters_xml.output parameters_hdf5.input parameters_hdf5.output
dist_noinst_DATA += parameters_mpi.ip-2 parameters_mpi.op-2

.PHONY : test tests example examples
test tests : check
example examples :

clean-local :
	rm -f parameterlist.dump parameters.dump
