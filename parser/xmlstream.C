/***************************************************************************
* ALPS++ library
*
* test/parser/xmlstream.C   test program for XML stream class
*
* $Id$
*
* Copyright (C) 2001-2003 by Synge Todo <wistaria@comp-phys.org>,
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

  oxs << alps::processing_instruction("xml-stylesheet")
      << alps::attribute("type", "text/xsl")
      << alps::attribute("href", "URL to my stylesheet");

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

      << x

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
}
