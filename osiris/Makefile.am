DEFAULT_INCLUDES = -I$(top_builddir)/src -I$(top_srcdir)/src
AM_CPPFLAGS = $(BASE_CPPFLAGS) $(BOOST_CPPFLAGS) $(XML_CPPFLAGS) $(HDF5_CPPFLAGS) $(MPI_CPPFLAGS)
AM_LDFLAGS = $(BASE_LDFLAGS) $(BOOST_LDFLAGS) $(SQLITE_LDFLAGS) $(XML_LDFLAGS) $(HDF5_LDFLAGS) $(MPI_LDFLAGS)
LDADD = $(top_builddir)/lib/alps/libalps.la $(BOOST_LIBS) $(SQLITE_LIBS) $(XML_LIBS) $(HDF5_LIBS) $(MPI_LIBS) $(BASE_LIBS) 

check_PROGRAMS = os sizeof xdrdump xdrdump2 boostdump boostdump2 boostdump3 boostdump4

os_SOURCES = os.C
sizeof_SOURCES = sizeof.C
xdrdump_SOURCES = xdrdump.C
xdrdump2_SOURCES = xdrdump2.C
boostdump_SOURCES = boostdump.C
boostdump2_SOURCES = boostdump2.C
boostdump3_SOURCES = boostdump3.C
boostdump4_SOURCES = boostdump4.C

dist_noinst_DATA = xdrdump.output xdrdump2.dump xdrdump2.output boostdump.output boostdump2.output boostdump3.output boostdump4.output

.PHONY : test tests example examples
test tests : check
example examples :

clean-local :
	rm -f xdrdump.dump textdump.dump
