/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#ifndef ALPS_PARAMS_PARAM_TYPES_RANKING_INCLUDED
#define ALPS_PARAMS_PARAM_TYPES_RANKING_INCLUDED

/** @file param_types_ranking.hpp Defines the convertability between types supported by alps::params.
 */


// MPL is used only to define integral constants, working around known compiler quirks
#include <boost/mpl/integral_c.hpp>

namespace alps {
    namespace params_ns {
        namespace detail {

            /* Overview:
               Each type is assigned a positive "rank" (negative means undefined rank). 
               Generally, types with a lower rank are "convertible" to types with a higher rank.
               Exceptions:
                 >> `char` and `bool` are not convertible to floating point;
                 >> `char` and `bool` are not interconvertible.

               NOTE: The above rules can be coded in a generic way, if we use
               something like `is_integral_type<T>` to distinguish the floating
               point types, and use `sizeof(T)` as the "rank".  However, there
               might be a disagreement regarding convertability between signed
               and unsigned types, so let's just code all supported types
               explicitly. (FIXME?)

            */
            
            template <typename>
            struct type_rank: public boost::mpl::integral_c<int,-1> {};

            template <>
            struct type_rank<bool> : public boost::mpl::integral_c<int,1> {};

            template <>
            struct type_rank<char> : public boost::mpl::integral_c<int,2> {};

            template <>
            struct type_rank<int> : public boost::mpl::integral_c<int,3> {};

            // `unsigned int` is convertible to and from `int` (FIXME?)
            template <>
            struct type_rank<unsigned int> : public boost::mpl::integral_c<int,3> {};

            // We declare `long` not convertible to `int` even if sizeof(int)==sizeof(long) (FIXME?)
            template <>
            struct type_rank<long> : public boost::mpl::integral_c<int,4> {};

            // `unsigned long` is convertible to and from `long` (FIXME?)
            template <>
            struct type_rank<unsigned long> : public boost::mpl::integral_c<int,4> {};

            template <>
            struct type_rank<double> : public boost::mpl::integral_c<int,5> {};

            // Generally, types are convertible if they have positive ranks and rank(FROM)<=rank(TO)
            template <typename FROM,typename TO>
            struct is_convertible
                : public boost::mpl::integral_c<bool,
                                                (type_rank<FROM>::value >=0) &&
                                                (type_rank<TO>::value >=0) &&
                                                (type_rank<FROM>::value <= type_rank<TO>::value) > {};

            // Generally, a type is always convertible to itself
            template <typename T>
            struct is_convertible<T,T> : public boost::mpl::integral_c<bool,true> {};

            // Specifically, char is not convertible to double
            template <>
            struct is_convertible<char,double> : public boost::mpl::integral_c<bool,false> {};

            // Specifically, bool is not convertible to double
            template <>
            struct is_convertible<bool,double> : public boost::mpl::integral_c<bool,false> {};

            // Specifically, bool is not convertible to char (and nothing else is convertible to char anyway, except char)
            template <>
            struct is_convertible<bool,char> : public boost::mpl::integral_c<bool,false> {};

        } // namespace detail
    } // namespace params_ns
} // namespace alps

          
#endif /* ALPS_PARAMS_PARAM_TYPES_RANKING_INCLUDED */
