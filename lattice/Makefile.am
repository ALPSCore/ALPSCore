DEFAULT_INCLUDES = -I$(top_builddir)/src -I$(top_srcdir)/src
AM_CPPFLAGS = $(BASE_CPPFLAGS) $(BOOST_CPPFLAGS) $(XML_CPPFLAGS) $(HDF5_CPPFLAGS) $(MPI_CPPFLAGS)
AM_LDFLAGS = $(BASE_LDFLAGS) $(BOOST_LDFLAGS) $(SQLITE_LDFLAGS) $(XML_LDFLAGS) $(HDF5_LDFLAGS) $(MPI_LDFLAGS)
LDADD = $(top_builddir)/lib/alps/libalps.la $(BOOST_LIBS) $(SQLITE_LIBS) $(XML_LIBS) $(HDF5_LIBS) $(MPI_LIBS) $(BASE_LIBS) 

check_PROGRAMS = example1 example2 example3 example4 example5 example6 example7 example8 example9 example10 example11 \
	coloring \
	label \
	parity

example1_SOURCES = example1.C
example2_SOURCES = example2.C
example3_SOURCES = example3.C
example4_SOURCES = example4.C
example5_SOURCES = example5.C
example6_SOURCES = example6.C
example7_SOURCES = example7.C
example8_SOURCES = example8.C
example9_SOURCES = example9.C
example10_SOURCES = example10.C
example11_SOURCES = example11.C
coloring_SOURCES = coloring.C
label_SOURCES = label.C
parity_SOURCES = parity.C

dist_noinst_DATA = example1.output example2.output example3.output example4.input example4.output example5.input example5.output example6.input example6.output example7.input example7.output example8.input example8.output example9.input example9.output example10.input example10.output example11.input example11.output \
	coloring.input coloring.output \
	label.input label.output \
	parity.input parity.output

.PHONY : test tests example examples
test tests : check
example examples :
