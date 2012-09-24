/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1997-2010 by Synge Todo <wistaria@comp-phys.org>
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

#include <alps/parapack/clone_info_p.h>
#include <alps/parser/xmlparser.h>
#include <boost/filesystem/operations.hpp>
#include <iostream>

int main()
{
#ifndef BOOST_NO_EXCEPTIONS
try {
#endif

  alps::clone_phase phase;
  alps::clone_phase_xml_handler handler(phase);

  alps::XMLParser parser(handler);
  parser.parse(std::cin);

  alps::oxstream ox(std::cout);

  ox << phase;

  boost::filesystem::path xdrpath("clone_phase.xdr");
  {
    alps::OXDRFileDump dp(xdrpath);
    dp << phase;
  }
  phase = alps::clone_phase();
  {
    alps::IXDRFileDump dp(xdrpath);
    dp >> phase;
  }
  ox << phase;
  boost::filesystem::remove(xdrpath);

  boost::filesystem::path h5path("clone_phase.h5");
  #pragma omp critical (hdf5io)
  {
    alps::hdf5::archive ar(h5path.string(), "a");
    ar["/phase"] << phase;
  }
  phase = alps::clone_phase();
  #pragma omp critical (hdf5io)
  {
    alps::hdf5::archive ar(h5path.string());
    ar["/phase"] >> phase;
  }
  ox << phase;
  boost::filesystem::remove(h5path);

#ifndef BOOST_NO_EXCEPTIONS
}
catch (std::exception& exp) {
  std::cerr << exp.what() << std::endl;
  std::abort();
}
#endif
  return 0;
}
