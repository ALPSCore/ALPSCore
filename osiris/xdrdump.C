/***************************************************************************
* ALPS++ library
*
* test/osiris/xdrdump.C   test program for xdrdump
*
* $Id$
*
* Copyright (C) 2001-2003 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
*                            Synge Todo <wistaria@comp-phys.org>,
*
* This program is free software; you can redistribute it and/or
* modify it under the terms of the GNU General Public License
* as published by the Free Software Foundation; either version 2
* of the License, or (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
**************************************************************************/

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
#ifndef BOOST_NO_INT64_t
  int64_t o8 = -1152921504606846976;
  uint64_t o9 = 18446744073709551614u;
#endif
  double o10 = 3.14159265358979323846;
  std::string o11 = "test string";

 {
   alps::OXDRFileDump od("xdrdump.dump");

   od << o1 << o2 << o3 << o4 << o5 << o6 << o7
#ifndef BOOST_NO_INT64_t
      << o8 << o9
#endif
      << o10 << o11;
 }

 alps::IXDRFileDump id("xdrdump.dump");
 std::cout << id.get<bool>() << ' ';
 std::cout << static_cast<int32_t>(id.get<int8_t>()) << ' ';
 std::cout << static_cast<int32_t>(id.get<uint8_t>()) << ' ';
 std::cout << id.get<int16_t>() << ' ';
 std::cout << id.get<uint16_t>() << ' ';
 std::cout << static_cast<int32_t>(id) << ' ';
 std::cout << static_cast<uint32_t>(id) << ' ';
#ifndef BOOST_NO_INT64_t
 int64_t i8 = id;
 uint64_t i9(id);
 std::cout << i8 << ' '  << i9  << ' ';
#endif
 double i10 = static_cast<double>(id);
 std::cout << i10 << ' ';
 std::string str;
 id >> str;
 std::cout << str << std::endl;

#ifndef BOOST_NO_EXCEPTIONS
}
catch (std::exception& exp) {
  std::cerr << exp.what() << std::endl;
  std::abort();
}
#endif
}
