/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2010 - 2011 by Lukas Gamper <gamperl@gmail.com>                   *
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


#ifndef ALPS_NGS_ALEA_ACCUMULATOR_HEADER
#define ALPS_NGS_ALEA_ACCUMULATOR_HEADER

#include <alps/ngs/alea/result.hpp>
#include <alps/ngs/alea/features.hpp>
#include <alps/ngs/alea/accumulator/arguments.hpp>
#include <alps/ngs/alea/accumulator/accumulator_impl.hpp>

#include <boost/parameter.hpp>

// = = = = N A M E D   P A R A M E T E R   C T O R   D E F I N I T I O N = = = =

namespace alps {
    namespace accumulator  {

        template<
              typename vt  = double
            , typename features_input = features<tag::mean, tag::error>
        >
        class accumulator: public detail::accumulator_impl<
              ValueType<vt>
            , typename features_input::_0
            , typename features_input::_1
            , typename features_input::_2
            , typename features_input::_3
            , typename features_input::_4
            , typename features_input::_5
            , typename features_input::_6
            , typename features_input::_7
            , typename features_input::_8
        > {
            typedef accumulator<vt, features_input> self_type;

            typedef detail::accumulator_impl<
                  ValueType<vt>
                , typename features_input::_0
                , typename features_input::_1
                , typename features_input::_2
                , typename features_input::_3
                , typename features_input::_4
                , typename features_input::_5
                , typename features_input::_6
                , typename features_input::_7
                , typename features_input::_8
            > base_type;

            typedef result<
                  ValueType<vt>
                , typename features_input::_0
                , typename features_input::_1
                , typename features_input::_2
                , typename features_input::_3
                , typename features_input::_4
                , typename features_input::_5
                , typename features_input::_6
                , typename features_input::_7
                , typename features_input::_8
            > result_type;
            
            public:
                accumulator(accumulator const & arg): base_type(static_cast<base_type const &>(arg)) {}
            
                BOOST_PARAMETER_CONSTRUCTOR(
                accumulator, 
                (base_type),
                keywords,
                    (optional 
                        (_bin_size, *)
                        (_bin_num, *)
                    )
                )

                result_type result() {
                    result_type result;
                    // TODO: implement
//                    base_type::make_result(result);
                    return result;
                }
        };
    } //end accumulator namespace 
} //end alps namespace
#endif // ALPS_NGS_ALEA_ACCUMULATOR_HEADER
