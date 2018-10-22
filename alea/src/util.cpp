/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#include <alps/alea/util.hpp>
#include <alps/alea/internal/format.hpp>

namespace alps { namespace alea {

std::ostream &operator<<(std::ostream &stream, verbosity verb)
{
    internal::get_format(stream, PRINT_TERSE) = verb;
    return stream;
}

}}
