/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_NGS_DETAIL_REMOVE_CVR_HPP
#define ALPS_NGS_DETAIL_REMOVE_CVR_HPP

namespace alps {
    namespace detail {

        template<typename T> struct remove_cvr {
            typedef T type;
        };
    
        template<typename T> struct remove_cvr<T const> {
            typedef typename remove_cvr<T>::type type;
        };
    
        template<typename T> struct remove_cvr<T volatile> {
            typedef typename remove_cvr<T>::type type;
        };
    
        template<typename T> struct remove_cvr<T &> {
            typedef typename remove_cvr<T>::type type;
        };

    }
}
#endif
