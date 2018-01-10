/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id: nobinning.h 3520 2009-12-11 16:49:53Z gamperl $ */


#pragma once

#include <cmath>
#include <boost/accumulators/numeric/functional/vector.hpp>
#include <boost/math/special_functions.hpp>


namespace alps {
  namespace numeric {
    
    // define special powers
    template<class T> 
    inline T sq(T value) {
        using boost::numeric::operators::operator*;
        return value * value; 
    }

    template<class T>
    inline T cb(T value) { 
        using boost::numeric::operators::operator*;
        return value * value * value; 
    }

    template<class T>
    inline T cbrt(T value) { 
        return std::pow(value,(T)(1./3.)); 
    }

    // define norm and r
    template <class T>
    inline T norm(T x, T y=T(), T z=T()) {
        using boost::numeric::operators::operator+;
        return (sq(x) + sq(y) + sq(z));
    }
    
    template <class T>
    inline T r(T x, T y=T(), T z=T()) {
        return std::sqrt(norm(x,y,z)); 
    }
  }
}
