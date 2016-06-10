/*
 * Copyright (C) 1998-2016 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file param_iterators.hpp Defines iterator types to go throw parameter lists.
*/

#ifndef ALPS_PARAMS_ITERATORS_27195b00f3bb43078545089e68f87bc5
#define ALPS_PARAMS_ITERATORS_27195b00f3bb43078545089e68f87bc5

#include <boost/iterator/filter_iterator.hpp>
#include <boost/iterator/transform_iterator.hpp>

#include "./option_type.hpp"

namespace alps {
    namespace params_ns {
        namespace detail {
            namespace iterators {
                
                // Functor to extract name from option map element
                class extract_option_name {
                  public:
                    typedef const std::string& result_type;
                    
                    const std::string& operator()(const options_map_type::value_type& opt) const
                    {
                        return opt.first;
                    }
                };

                // Functor to check if option map element contains a missing option
                class has_option_missing {
                  public:
                    bool operator()(const options_map_type::value_type& opt) const
                    {
                        return is_option_missing(opt.second);
                    }
                };

                // Internal type: Iterator through "missing" options (names+values)
                typedef boost::filter_iterator<has_option_missing,
                                               options_map_type::const_iterator> missing_pairs_iterator;
            
                // Iterator through names of "missing" (defined, but unassigned) parameters
                typedef boost::transform_iterator<extract_option_name,
                                                  missing_pairs_iterator> missing_params_iterator;


                // Makes an iterator using given start and end base iterators
                inline missing_params_iterator make_missing_params_iterator(
                    const options_map_type::const_iterator& begin,
                    const options_map_type::const_iterator& end)
                {
                    return missing_params_iterator(missing_pairs_iterator(begin,end));
                }
                
            } // iterators::
        } // detail::
    } // params_ns::
} // alps::

#endif /* ALPS_PARAMS_ITERATORS_27195b00f3bb43078545089e68f87bc5 */

