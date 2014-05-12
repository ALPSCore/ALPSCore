/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2006-2009 by Synge Todo <wistaria@comp-phys.org>
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

#include <alps/parameter/parameter.h>
#include <boost/throw_exception.hpp>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

int main()
{
#ifndef BOOST_NO_EXCEPTIONS
  try {
#endif

#ifndef BOOST_MSVC
  setenv("DIR", "/home/alps", 1);
#else
  _putenv("DIR=/home/alps");
#endif

  std::string str;
  alps::Parameter p;
  while (std::getline(std::cin, str)) {
#ifndef BOOST_NO_EXCEPTIONS
    try {
#endif
    p.parse(str);
    str = p.value().c_str();
    std::cout << p.key() << " = ";
    if (str.find(' ') != std::string::npos)
      std::cout << '"' << str << '"';
    else
      std::cout << str;
    std::cout << ";\n";
#ifndef BOOST_NO_EXCEPTIONS
    }
    catch (std::exception& e) {
      std::cout << "Caught exception: " << e.what() << "\n";
    }
#endif
  }

#ifndef BOOST_NO_EXCEPTIONS
}
catch (std::exception& e) {
  std::cerr << "Caught exception: " << e.what() << "\n";
  exit(-1);
}
catch (...) {
  std::cerr << "Caught unknown exception\n";
  exit(-2);
}
#endif
  return 0;
}
