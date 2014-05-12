/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2001-2006 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
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

#include <alps/expression.h>

#include <boost/throw_exception.hpp>
#include <iostream>
#include <stdexcept>

int main()
{
#ifndef BOOST_NO_EXCEPTIONS
  try {
#endif

  alps::Parameters parms;
  std::string str;
  while (std::getline(std::cin, str) && str.size() && str[0] != '%')
    parms.push_back(alps::Parameter(str));
  std::cout << "Parameters:\n" << parms << std::endl;
  
  alps::ParameterEvaluator eval(parms);
  while (std::cin) {
    alps::Expression expr(std::cin);
    if (!expr.can_evaluate(eval))
      std::cout << "Cannot evaluate [" << expr << "]." << std::endl;
    else 
      std::cout << "The value of [" << expr << "] is "
                << alps::evaluate<double>(expr, eval) << std::endl;
    char c;
    std::cin >> c;
    if (c!=',')
      break;
  }

  while (std::cin) {
    std::string v;
    std::cin >> v;
    if (v.empty()) break;
    if (!alps::can_evaluate(v, parms))
      std::cout << "Cannot evaluate [" << v << "]." << std::endl;
    else 
      std::cout << "The value of [" << v << "] is "
                << alps::evaluate<double>(v, parms) << std::endl;
  }

#ifndef BOOST_NO_EXCEPTIONS
}
catch (std::exception& e)
{
  std::cerr << "Caught exception: " << e.what() << "\n";
  exit(-1);
}
catch (...)
{
  std::cerr << "Caught unknown exception\n";
  exit(-2);
}
#endif
  return 0;
}
