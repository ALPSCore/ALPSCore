/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2001-2003 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
*                            Synge Todo <wistaria@comp-phys.org>
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

/* $Id$ */

#include <alps/osiris.h>
#include <iostream>
#include <cstdlib>

int main()
{
#ifndef BOOST_NO_EXCEPTIONS
try {
#endif

  bool o1 = false;
  int8_t o2 = 63;
  uint8_t o3 = 201;
  int16_t o4 = -699;
  uint16_t o5 = 43299;
  int32_t o6 = 847229;
  uint32_t o7 = 4294967295u;
  int64_t o8 = -1152921504606846976ll;
  int64_t o9 = 343434545665ll;
  uint64_t o10 = 18446744073709551614ull;
  double o11 = 3.14159265358979323846;
  std::string o12 = "test string";
  std::complex<double> o13(1,2);

  {
    alps::OXDRFileDump od(boost::filesystem::path("xdrdump.dump"));
    od << o1 << o2 << o3 << o4 << o5 << o6 << o7 << o8 << o9 << o10 << o11 << o12 <<o13;
  }
  
  alps::IXDRFileDump id(boost::filesystem::path("xdrdump.dump"));
  std::cout << id.get<bool>() << ' ';
  std::cout << static_cast<int32_t>(id.get<int8_t>()) << ' ';
  std::cout << static_cast<int32_t>(id.get<uint8_t>()) << ' ';
  std::cout << id.get<int16_t>() << ' ';
  std::cout << id.get<uint16_t>() << ' ';
  std::cout << static_cast<int32_t>(id) << ' ';
  std::cout << static_cast<uint32_t>(id) << ' ';
  int64_t i8 = id;
  int64_t i9 = id;
  uint64_t i10(id);
  std::cout << i8 << ' '  << i9  << ' ' << i10 << ' ';
  double i11 = static_cast<double>(id);
  std::cout << i11 << ' ';
  std::string str;
  id >> str;
  std::cout << str << ' ';
  std::complex<double> i13;
  id >> i13;
  std::cout << i13 << std::endl;
  
#ifndef BOOST_NO_EXCEPTIONS
}
catch (std::exception& exp) {
  std::cerr << exp.what() << std::endl;
  std::abort();
}
#endif
  return 0;
}
