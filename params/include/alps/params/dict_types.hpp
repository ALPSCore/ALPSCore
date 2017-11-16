/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_PARAMS_PARAM_TYPES_HPP_2b33f1b375e64b6fa9adcb68d7de2407
#define ALPS_PARAMS_PARAM_TYPES_HPP_2b33f1b375e64b6fa9adcb68d7de2407

#include <vector>
#include <string>
#include <stdexcept>
#include <boost/mpl/vector/vector10.hpp>
#include <boost/mpl/transform.hpp>
#include <boost/mpl/front_inserter.hpp>
#include <boost/mpl/copy.hpp>

// forward declarations
namespace alps{ namespace hdf5 {
    class archive;
}}

namespace alps {
    namespace params_ns {
        namespace detail {
            
            // Have namespaces handy
            namespace mpl=::boost::mpl;
            namespace mplh=::boost::mpl::placeholders;

            /// "Empty value" type
            struct None {
                void save(alps::hdf5::archive&) const { throw std::logic_error("None::save() should never be called"); }
                void load(alps::hdf5::archive&) { throw std::logic_error("None::load() should never be called"); }
            };
            template <typename S>
            inline S& operator<<(S&, const None&) { throw std::logic_error("Generic streaming operator of None should never be called"); }

            // List of allowed basic scalar types:
            typedef mpl::vector8<bool,
                                 int,
                                 unsigned int,
                                 long int,
                                 unsigned long int,
                                 float,
                                 double,
                                 std::string> dict_scalar_types;
            
            // // List of allowed pairs:  (removed until clarification)
            // typedef mpl::transform< dict_scalar_types, std::pair<std::string, mplh::_1> >::type dict_pair_types;

            // List of allowed vectors:
            typedef mpl::transform< dict_scalar_types, std::vector<mplh::_1> >::type dict_vector_types;

            // Meta-function returning a new sequence (FS, TS) from sequences FS and TS
            template <typename FS, typename TS>
            struct copy_to_front : public mpl::reverse_copy< FS, mpl::front_inserter<TS> > {};

            typedef mpl::push_front<
                copy_to_front<dict_scalar_types, dict_vector_types>::type,
                None
                >::type dict_all_types;
            // This list includes std::pair, removed until clarification
            // typedef mpl::push_front<
            //     copy_to_front<dict_scalar_types, 
            //                   copy_to_front<dict_vector_types, dict_pair_types>::type
            //                  >::type,
            //     None
            //     >::type dict_all_types;
            // List of allowed types, `None` being the first

            
        } // ::detail
    } // ::params_ns
}// ::alps


#endif /* ALPS_PARAMS_PARAM_TYPES_HPP_2b33f1b375e64b6fa9adcb68d7de2407 */
