/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#ifndef ALPS_NUMERIC_MATRIX_DETAIL_DEBUG_OUTPUT_HPP
#define ALPS_NUMERIC_MATRIX_DETAIL_DEBUG_OUTPUT_HPP

#ifdef ALPS_NUMERIC_MATRIX_DEBUG
#include <typeinfo>
#include <iostream>
#define ALPS_NUMERIC_MATRIX_DEBUG_OUTPUT(T) \
    std::cerr << __FILE__ <<" " << __LINE__ << ":" << T << std::endl;
#else
#define ALPS_NUMERIC_MATRIX_DEBUG_OUTPUT(T)
#endif

#endif //ALPS_NUMERIC_MATRIX_DETAIL_DEBUG_OUTPUT_HPP
