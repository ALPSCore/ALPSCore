/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2011 - 2012 by Mario Koenz <mkoenz@ethz.ch>                       *
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

#ifndef ALPS_NGS_ALEA_ACCUMULATOR_IMPL_HEADER
#define ALPS_NGS_ALEA_ACCUMULATOR_IMPL_HEADER

#include <alps/ngs/alea/detail/accumulator_detail.hpp>

#include <boost/parameter.hpp>

// = = = = N A M E D   P A R A M E T E R   C T O R   D E F I N I T I O N = = = =

namespace alps
{
    namespace alea
    {
        //TODO: put the parameter in a namespace
        BOOST_PARAMETER_NAME((bin_size, keywords) _bin_size)
        BOOST_PARAMETER_NAME((bin_number, keywords) _bin_number)
        BOOST_PARAMETER_NAME((bin_log, keywords) _bin_log)

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
        > 
        class accumulator: public detail::accumulator_impl<ValueType<_0>, _1, _2, _3, _4, _5, _6, _7, _8, _9>
        {
            typedef detail::accumulator_impl<ValueType<_0>, _1, _2, _3, _4, _5, _6, _7, _8, _9> base;
            
            public:
                accumulator(accumulator const & arg): base(static_cast<base const &>(arg)) {}
            
                BOOST_PARAMETER_CONSTRUCTOR(
                accumulator, 
                (base),
                keywords,
                    (optional 
                        (_bin_size, *)
                        (_bin_number, *)
                        (_bin_log, *)
                    )
                )
        };
    }//end alea namespace 
}//end alps namespace
#endif // ALPS_NGS_ALEA_ACCUMULATOR_IMPL_HEADER
