/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1997-2009 by Synge Todo <wistaria@comp-phys.org>
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

#include "util.h"
#include <alps/random/pseudo_des.h>
#include <boost/classic_spirit.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/throw_exception.hpp>
#include <stdexcept>

namespace alps {

int hash(int n, int s) {
  static const unsigned int hash_seed = 3777549;
  return (alps::pseudo_des::hash(s, n ^ hash_seed) ^ alps::pseudo_des::hash(s, hash_seed)) &
    ((1<<30) | ((1<<30)-1));
}

std::string id2string(int id, std::string const& pad) {
  int i = id;
  std::string str;
  while (i >= 10) {
    str += pad;
    i /= 10;
  }
  str += boost::lexical_cast<std::string>(id);
  return str;
}

double parse_percentage(std::string const& str) {
  using namespace boost::spirit;
  double r;
  if (!parse(str.c_str(), real_p[assign_a(r)] >> '%' >> end_p, space_p).full)
    boost::throw_exception(std::runtime_error("error in parsing \"" + str + '\"'));
  return 0.01 * r;
}

} // end namespace alps
