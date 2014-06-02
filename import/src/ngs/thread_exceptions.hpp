/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_NGS_THREAD_INTERRUPTED_HPP
#define ALPS_NGS_THREAD_INTERRUPTED_HPP

#ifdef ALPS_NGS_SINGLE_THREAD

namespace boost {

    struct thread_interrupted {};

    struct lock_error : public std::logic_error {};

}

#else

#include <boost/thread/exceptions.hpp>

#endif

#endif
