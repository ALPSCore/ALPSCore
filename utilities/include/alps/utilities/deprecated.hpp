/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_UTILITIES_DEPRECATED_HPP
#define ALPS_UTILITIES_DEPRECATED_HPP

// FIXME: Test for other compilers here
#if __GNUG__
#define ALPS_DEPRECATED __attribute__((deprecated))
#else
#define ALPS_DEPRECATED
#endif

#define ALPS_STRINGIFY(arg) ALPS_STRINGIFY_HELPER(arg)

#define ALPS_STRINGIFY_HELPER(arg) #arg

#endif /* ALPS_UTILITIES_DEPRECATED_HPP */
