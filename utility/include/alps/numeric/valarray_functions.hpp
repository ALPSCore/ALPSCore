/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id: valarray_functions.h 3520 2009-12-11 16:49:53Z tamama $ */

#ifndef ALPS_NUMERIC_VALARRAY_FUNCTIONS_HPP
#define ALPS_NUMERIC_VALARRAY_FUNCTIONS_HPP



namespace alps { 
  namespace numeric {

    template <class T>
    std::ostream& operator<< (std::ostream &out, std::valarray<T> const & val)
    {
      std::copy(&const_cast<std::valarray<T>&>(val)[0],&const_cast<std::valarray<T>&>(val)[0]+val.size(),std::ostream_iterator<T>(out,"\t"));
      return out;
    }


  }
}

#endif // ALPS_NUMERIC_VALARRAY_FUNCTIONS_HPP




