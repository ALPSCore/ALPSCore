/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_PARAMS_PARAM_TYPES_HPP_2b33f1b375e64b6fa9adcb68d7de2407
#define ALPS_PARAMS_PARAM_TYPES_HPP_2b33f1b375e64b6fa9adcb68d7de2407

#include <vector>
#include <string>
#include <boost/mpl/list/list10.hpp>
#include <boost/mpl/transform.hpp>
#include <boost/mpl/front_inserter.hpp>
#include <boost/mpl/copy.hpp>


namespace alps {
    namespace params_new_ns {
        namespace detail {
            
            // Have namespaces handy
            namespace mpl=::boost::mpl;
            namespace mplh=::boost::mpl::placeholders;

            /// "Empty value" type
            struct None {};

            // List of allowed basic scalar types:
            typedef mpl::list8<bool,
                               int,
                               unsigned int,
                               long int,
                               unsigned long int,
                               float,
                               double,
                               std::string> dict_scalar_types;
            
            // List of allowed pairs:
            typedef mpl::transform< dict_scalar_types, std::pair<std::string, mplh::_1> >::type dict_pair_types;

            // List of allowed vectors:
            typedef mpl::transform< dict_scalar_types, std::vector<mplh::_1> >::type dict_vector_types;

            // Meta-function returning a new sequence (FS, TS) from sequences FS and TS
            template <typename FS, typename TS>
            struct copy_to_front : public mpl::reverse_copy< FS, mpl::front_inserter<TS> > {};

            // List of allowed types, `None` being the first
            typedef mpl::push_front<
                copy_to_front<dict_scalar_types, 
                              copy_to_front<dict_vector_types, dict_pair_types>::type
                             >::type,
                None
                >::type dict_all_types;
            
        } // detail
    } // params_new_ns
}// alps


#endif /* ALPS_PARAMS_PARAM_TYPES_HPP_2b33f1b375e64b6fa9adcb68d7de2407 */
