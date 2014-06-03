/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_NGS_MADE_DEPRECATED_PARAMETERS_HPP
#define ALPS_NGS_MADE_DEPRECATED_PARAMETERS_HPP

#include <alps/ngs/params.hpp>

#include <alps/parameter.h>
#include <alps/config.h>

namespace alps {

    ALPS_DECL params make_parameters_from_xml(boost::filesystem::path const & filename);

}

#endif
