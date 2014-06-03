/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

/// \file cctype.h
/// \brief A safe version of the standard cctype header
///
///  Some cctype headers do not undefine harmful macros, so undefine
///  them here.

#ifndef ALPS_CCTYPE_H
#define ALPS_CCTYPE_H

#include <cctype>

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
