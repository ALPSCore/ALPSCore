/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id: nobinning.h 3520 2009-12-11 16:49:53Z gamperl $ */


#include <boost/python.hpp>
#include <alps/scheduler/convert.h>
#include <alps/utilities/encode.hpp>
#include <alps/random.h>
#include <alps/parser/xslt_path.h>
#include <alps/python/make_copy.hpp>
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include<boost/random/uniform_01.hpp>

typedef boost::variate_generator<boost::mt19937, boost::uniform_01<double> > random_01;

class WrappedRNG : public random_01
{
public:
    WrappedRNG(int seed=0)
    : random_01(boost::mt19937(seed), boost::uniform_01<double>())
    {
    }
};

const char convert2xml_docstring[] = 
  "converts a file to XML\n"
  "\n"
  "This function takes the path to an ALPS file as input and converts it to XML.\n"
  "It returns a string with the path to the resulting XML file";

const char hdf5_name_encode_docstring[] = 
"encodes a string for use in HDF5 paths\n"
"\n"
"This function takes a string and escapes all needed characters for it to be "
"used in HDF5 path names.";

const char hdf5_name_decode_docstring[] = 
"decodes a string fromHDF5 paths\n"
"\n"
"This function takes a string used in an HDF5 path name and replaces all "
"escaped characters.";

const char search_xml_library_path_docstring[] = 
"returns the full path for an ALPS XML file\n"
"\n"
"This function takes the name for an ALPS library XML or XSL file and returns "
"the full path.";

const char rng_docstring[] = 
"a uniform random number generator class\n\n"
"This class uses the Mersenne Twister rgenerator mt19937 to generate uniform "
"random numbers in the range [0,1).\n"
"The constructor takes an optional integer random seed argument.\n"
"Random numbers are created using the function call operator.\n";


namespace  {
  void wrap_with_signature()
  {
    using namespace boost::python;
    def("convert2xml", alps::convert2xml,convert2xml_docstring);
    def("hdf5_name_encode", alps::hdf5_name_encode,hdf5_name_encode_docstring);
    def("hdf5_name_decode", alps::hdf5_name_decode,hdf5_name_decode_docstring);
    def("search_xml_library_path", alps::search_xml_library_path,search_xml_library_path_docstring);
    /*
     def("convert2numpy",
     static_cast<boost::python::numeric::array(*)(std::vector<double> const& )>
     (&convert2numpy));
     def("convert2numpy",
     static_cast<boost::python::numeric::array(*)(std::vector<int> const& )>
     (&convert2numpy));
     
     def("convert2vector",&convert2vector<double>);
     def("convert2vector",&convert2vector<int>);
     */
    
  }
  
  void wrap_without_signature()
  {
    using namespace boost::python;
    docstring_options doc_options(true);
    doc_options.disable_cpp_signatures();
    class_<WrappedRNG>("rng", rng_docstring,init<optional<int> >("the constructor takes an optional integer argument as random number seed"))
    .def("__deepcopy__",  &alps::python::make_copy<WrappedRNG>, "the deepcopy function creates a new copy of the generator")
    .def("__call__", static_cast<WrappedRNG::result_type(WrappedRNG::*)()>(&WrappedRNG::operator()), "returns a uniform random number in [0,1)")
    ;
  }
  
}

BOOST_PYTHON_MODULE(pytools_c)
{
  using namespace boost::python;
  wrap_with_signature();
  wrap_without_signature();
}


