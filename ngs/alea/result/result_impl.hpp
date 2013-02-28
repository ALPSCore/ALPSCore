/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2011 - 2012 by Lukas Gamper <gamperl@gmail.com>                   *
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

#ifndef ALPS_NGS_ALEA_RESULT_RESULT_IMPL_HEADER
#define ALPS_NGS_ALEA_RESULT_RESULT_IMPL_HEADER

#include <alps/ngs/alea/features.hpp>

#include <boost/static_assert.hpp>
#include <boost/type_traits.hpp>
#include <boost/utility.hpp>

#include <iostream>

namespace alps {
    namespace accumulator {
        namespace detail {

        // = = = = = = = A C C U M U L A T O R _ I M P L= = = = = = = = = =
            
            template<
                  typename _0  = void
                , typename _1  = void
                , typename _2  = void
                , typename _3  = void
                , typename _4  = void
                , typename _5  = void
                , typename _6  = void
                , typename _7  = void
                , typename _8  = void
                , typename _9  = void
            >  struct result_impl : public DeriveResultProperties<
                  typename UniqueList<typename ResolveDependencies<typename ValueTypeFirst<typename MakeList<
                      _0, _1, _2, _3, _4, _5, _6, _7, _8, _9
                  >::type>::type>::type>::type
                , UselessBase
            >::type {
                //typename it for shorter syntax
                typedef typename DeriveResultProperties<
                      typename UniqueList<typename ResolveDependencies<typename ValueTypeFirst<typename MakeList<
                          _0, _1, _2, _3, _4, _5, _6, _7, _8, _9
                      >::type>::type>::type>::type
                    , UselessBase
                >::type base_type;

                result_impl() {}
                result_impl(result_impl const & arg): base_type(arg) {}
            };
            
        // = = = = = = S T R E A M   O P E R A T O R = = = = = = = = = = =
            template<
                  typename _0
                , typename _1
                , typename _2
                , typename _3
                , typename _4
                , typename _5
                , typename _6
                , typename _7
                , typename _8
                , typename _9
            > inline std::ostream & operator <<(std::ostream & os, result_impl<_0, _1, _2, _3, _4, _5, _6, _7, _8, _9> & a) {
                a.print(os);
                return os;
            }
        }
    }
}

#endif
