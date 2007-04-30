SUBDIRS = fixed_capacity osiris parameter parser
if ALPS_FULL
SUBDIRS += alea lattice model
endif
SUBDIRS += .
DIST_SUBDIRS = $(SUBDIRS)

TESTS_ENVIRONMENT = $(top_srcdir)/config/run-test $(srcdir)

TESTS_FC = fixed_capacity/fixed_capacity_deque fixed_capacity/fixed_capacity_traits fixed_capacity/fixed_capacity_vector fixed_capacity/test_deque fixed_capacity/test_vector
TESTS_OSIRIS = osiris/sizeof osiris/xdrdump osiris/xdrdump2 osiris/boostdump osiris/boostdump2 osiris/boostdump3 osiris/boostdump4
TESTS_PARAMETER = parameter/expression parameter/expression2 parameter/flatten parameter/parameter parameter/parameterlist parameter/parameterlist_xml parameter/parameters parameter/parameters_xml
TESTS_PARSER = parser/xmlhandler parser/xmlparser parser/xmlstream
TESTS_ALEA = alea/detailedbinning alea/observableset_xml alea/signed alea/simpleobseval alea/testobservableset alea/vectorobseval
TESTS_LATTICE = lattice/example1 lattice/example2 lattice/example3 lattice/example4 lattice/example5 lattice/example6 lattice/example7 lattice/example8 lattice/example9 lattice/example10 \
	lattice/label \
	lattice/parity
TESTS_MODEL = model/example1 model/example2 model/example3 model/example4 model/example5 model/example6 model/example7 model/example8 model/example9 model/example10 model/example11 model/example12 model/example13 model/example14 model/example15 model/example16 model/example17 model/example18

TESTS = $(TESTS_FC) $(TESTS_OSIRIS) $(TESTS_PARAMETER) $(TESTS_PARSER)
if ALPS_FULL
TESTS += $(TESTS_ALEA) $(TESTS_LATTICE) $(TESTS_MODEL)
endif

.PHONY : test tests example examples
test tests : check
example examples :
