/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
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

#endif /* ALPS_UTILITIES_DEPRECATED_HPP */
