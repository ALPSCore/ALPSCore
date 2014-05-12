/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2001-2006 by Synge Todo <wistaria@comp-phys.org>
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

#include <alps/parser/xmlstream.h>

#include <cstdlib>
#include <stdexcept>

int main()
{
#ifndef BOOST_NO_EXCEPTIONS
try {
#endif

  double x = 3.14;

  alps::oxstream oxs;

  oxs << alps::header("MyEncoding");

  oxs << alps::stylesheet("URL to my stylesheet")
      << alps::processing_instruction("my_pi");

  oxs << alps::start_tag("tag0")
      << alps::attribute("name0", 1)

      << "this is a text"

      << alps::start_tag("tag1")
      << alps::start_tag("tag2")
      << alps::xml_namespace("MyNameSpace", "MyURL")
    
      << "text 2 "
      << "text 3 " << std::endl
      << alps::precision(3.14159265358979323846, 3) << ' '
      << alps::precision(3.14159265358979323846, 6) << '\n'
      << "text 4" << std::endl
      << alps::convert("text <&\">'")

      << alps::start_tag("tag3")
      << alps::end_tag

      << alps::precision(x, 6)

      << alps::start_tag("tag4") << alps::no_linebreak
      << "no linebreak"
      << alps::end_tag

      << alps::end_tag("tag2")
      << alps::end_tag("tag1")
      << alps::end_tag;

#ifndef BOOST_NO_EXCEPTIONS
}
catch (std::exception& exp) {
  std::cerr << exp.what() << std::endl;
  std::abort();
}
#endif
  return 0;
}
