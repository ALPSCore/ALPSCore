DEFAULT_INCLUDES = -I$(top_builddir)/src -I$(top_srcdir)/src
AM_CPPFLAGS = $(BASE_CPPFLAGS) $(BOOST_CPPFLAGS) $(XML_CPPFLAGS) $(HDF5_CPPFLAGS)
AM_LDFLAGS = $(BASE_LDFLAGS) $(BOOST_LDFLAGS) $(HDF5_LDFLAGS) $(XML_LDFLAGS)
LDADD = $(top_builddir)/lib/alps/libalps.la $(top_builddir)/lib/comm-sgl/libcomm-sgl.la $(BOOST_LIBS) $(HDF5_LIBS) $(XML_LIBS) $(BASE_LIBS)

check_PROGRAMS = fixed_capacity_deque fixed_capacity_traits fixed_capacity_vector test_deque test_vector timing_queue timing_stack timing_vector

fixed_capacity_deque_SOURCES = fixed_capacity_deque.C
fixed_capacity_traits_SOURCES = fixed_capacity_traits.C
fixed_capacity_vector_SOURCES = fixed_capacity_vector.C
test_deque_SOURCES = test_deque.C
test_vector_SOURCES = test_vector.C
timing_queue_SOURCES = timing_queue.C
timing_stack_SOURCES = timing_stack.C
timing_vector_SOURCES = timing_vector.C

noinst_HEADERS = test_main.h
dist_noinst_DATA = fixed_capacity_deque.output fixed_capacity_traits.output fixed_capacity_vector.output test_deque.output test_vector.output

.PHONY : test tests example examples
test tests : check
example examples :
