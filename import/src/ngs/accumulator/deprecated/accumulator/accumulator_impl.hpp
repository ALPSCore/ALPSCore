/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_NGS_ALEA_ACCUMULATOR_ACCUMULATOR_IMPL_HEADER
#define ALPS_NGS_ALEA_ACCUMULATOR_ACCUMULATOR_IMPL_HEADER

#include <alps/ngs/alea/features.hpp>

#include <boost/static_assert.hpp>
#include <boost/parameter.hpp>
#include <boost/type_traits.hpp>
#include <boost/utility.hpp>

#include <iostream>

namespace alps {
    namespace accumulator {
        namespace detail {

        // = = = = = = = A C C U M U L A T O R _ I M P L= = = = = = = = = =
            
            template<
                  typename A0  = void
                , typename A1  = void
                , typename A2  = void
                , typename A3  = void
                , typename A4  = void
                , typename A5  = void
                , typename A6  = void
                , typename A7  = void
                , typename A8  = void
                , typename A9  = void
                , typename A10  = void
                , typename A11  = void
                , typename A12  = void
                , typename A13  = void
                , typename A14  = void
                , typename A15  = void
                , typename A16  = void
                , typename A17  = void
                , typename A18  = void
                , typename A19  = void
                , typename A20  = void
                , typename A21  = void
            >  struct accumulator_impl : public DeriveAccumulatorProperties<
                  typename UniqueList<typename ResolveDependencies<typename TypeHolderFirst<typename MakeList<
                      A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21
                  >::type>::type>::type>::type
                , UselessBase
            >::type {
                //typename it for shorter syntax
                typedef typename DeriveAccumulatorProperties<
                      typename UniqueList<typename ResolveDependencies<typename TypeHolderFirst<typename MakeList<
                          A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21
                      >::type>::type>::type>::type
                    , UselessBase
                >::type base_type;

                //disable_if is required bc of the named parameter. This template shouldn't act as a copy-ctor
                template <typename ArgumentPack> accumulator_impl(
                      ArgumentPack const & args
                    , typename boost::disable_if<boost::is_base_of<accumulator_impl, ArgumentPack>, int>::type = 0
                ): base_type(args) {}

                //copy-ctor
                accumulator_impl(accumulator_impl const & arg): base_type(arg) {}
            };
            
        // = = = = = = S T R E A M   O P E R A T O R = = = = = = = = = = =
            template<
                  typename A0
                , typename A1
                , typename A2
                , typename A3
                , typename A4
                , typename A5
                , typename A6
                , typename A7
                , typename A8
                , typename A9
                , typename A10
                , typename A11
                , typename A12
                , typename A13
                , typename A14
                , typename A15
                , typename A16
                , typename A17
                , typename A18
                , typename A19
                , typename A20
                , typename A21
            > inline std::ostream & operator <<(std::ostream & os, accumulator_impl<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21> & acc) {
                acc.print(os);
                return os;
            }
        } // end namespace detail
    } // end accumulator namespace 
} // end alps namespace

#endif // ALPS_NGS_ALEA_ACCUMULATOR_ACCUMULATOR_IMPL_HEADER
