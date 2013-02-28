/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2011 - 2012 by Lukas Gamper <gamperl@gmail.com>                   *
 *                              Mario Koenz <mkoenz@ethz.ch>                       *
 *                                                                                 *
 * This software is part of the ALPS libraries, published under the ALPS           *
 * Library License; you can use, redistribute it and/or modify it under            *
 * the terms of the license, either version 1 or (at your option) any later        *
 * version.                                                                        *
 *                                                                                 *
 * You should have received a copy of the ALPS Library License along with          *
 * the ALPS Libraries; see the file LICENSE.txt. If not, the license is also       *
 * available from http://alps.comp-phys.org/.                                      *
 *                                                                                 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,        *
 * FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT       *
 * SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE       *
 * FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,     *
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER     *
 * DEALINGS IN THE SOFTWARE.                                                       *
 *                                                                                 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

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
            >  struct accumulator_impl : public DeriveAccumulatorProperties<
                  typename UniqueList<typename ResolveDependencies<typename ValueTypeFirst<typename MakeList<
                      A0, A1, A2, A3, A4, A5, A6, A7, A8, A9
                  >::type>::type>::type>::type
                , UselessBase
            >::type {
                //typename it for shorter syntax
                typedef typename DeriveAccumulatorProperties<
                      typename UniqueList<typename ResolveDependencies<typename ValueTypeFirst<typename MakeList<
                          A0, A1, A2, A3, A4, A5, A6, A7, A8, A9
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
            > inline std::ostream & operator <<(std::ostream & os, accumulator_impl<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9> & a) {
                a.print(os);
                return os;
            }
        } // end namespace detail
    } // end accumulator namespace 
} // end alps namespace

#endif // ALPS_NGS_ALEA_ACCUMULATOR_ACCUMULATOR_IMPL_HEADER
