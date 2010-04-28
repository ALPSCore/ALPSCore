/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2009 by Ping Nang Ma <pingnang@itp.phys.ethz.ch>,
*                            Matthias Troyer <troyer@itp.phys.ethz.ch>,
*                            Bela Bauer <bauerb@itp.phys.ethz.ch>
*
* This software is part of the ALPS libraries, published under the ALPS
* Library License; you can use, redistribute it and/or modify it under
* the terms of the license, either version 1 or (at your option) any later
* version.
*
* You should have received a copy of the ALPS Library License along with
* the ALPS Libraries; see the file LICENSE.txt. If not, the license is also
* available from http://alps.comp-phys.org/.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
* SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
* FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
* ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*
*****************************************************************************/

/* $Id: nobinning.h 3520 2009-12-11 16:49:53Z gamperl $ */


#include <boost/python.hpp>
#include <alps/scheduler/convert.h>
#include <alps/utility/encode.hpp>
#include <alps/parser/xslt_path.h>

#include <alps/python/make_copy.hpp>
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include <boost/random.hpp>
#include<boost/random/uniform_01.hpp>

typedef boost::variate_generator<boost::mt19937&, boost::uniform_01<double> > random_01;

class WrappedRNG : public random_01
{
public:
    WrappedRNG(int seed=0)
    :eng(seed), dist(), random_01(eng, dist)
    {
    }
    unsigned int seed;
    boost::mt19937 eng;
    boost::uniform_01<double> dist;
    
};

BOOST_PYTHON_MODULE(pytools)
{
  using namespace boost::python;
  def("convert2xml", alps::convert2xml);
  def("hdf5_name_encode", alps::hdf5_name_encode);
  def("hdf5_name_decode", alps::hdf5_name_decode);
  def("search_xml_library_path", alps::search_xml_library_path);
    
    class_<WrappedRNG>("rng", init<optional<int> >() )
    .def("__deepcopy__",  &alps::python::make_copy<WrappedRNG>)
    .def("__call__", static_cast<WrappedRNG::result_type(WrappedRNG::*)()>(&WrappedRNG::operator()))
    ;
}

