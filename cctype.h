/***************************************************************************
* ALPS++ library
*
* alps/cctype.h
*
* $Id$
*
* Copyright (C) 1994-2002 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
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

#ifndef ALPS_CCTYPE_H
#define ALPS_CCTYPE_H

#include <cctype>

//
// Some <cctype> header does not undefine harmful macros, so undefine
// them.
//

#ifdef isspace 
# undef isspace
#endif
#ifdef isprint
# undef isprint
#endif
#ifdef iscntrl
# undef iscntrl
#endif
#ifdef isupper
# undef isupper
#endif
#ifdef islower
# undef islower
#endif
#ifdef isalpha
# undef isalpha
#endif
#ifdef isdigit
# undef isdigit
#endif
#ifdef ispunct
# undef ispunct
#endif
#ifdef isxdigit
# undef isxdigit
#endif
#ifdef isalnum
# undef isalnum
#endif
#ifdef isgraph
# undef isgraph
#endif
#ifdef toupper
# undef toupper
#endif
#ifdef tolower
# undef tolower
#endif

#endif // ALPS_CCTYPE_H
