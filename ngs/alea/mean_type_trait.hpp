/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2011 - 2012 by Mario Koenz <mkoenz@ethz.ch>                       *
 *                      2012 by Lukas Gamper <gamperl@gmail.com>                   *
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


#ifndef ALPS_NGS_ALEA_MEAN_TRAIT_HEADER
#define ALPS_NGS_ALEA_MEAN_TRAIT_HEADER

#include <boost/type_traits.hpp>
#include <boost/static_assert.hpp>

//= = = = = = = = = = = = = = = M E A N   T R A I T = = = = = = = = = = = = = = = =
namespace alps
{
    namespace alea
    {
        namespace detail
        {
            template<unsigned n> struct static_array
            {
                char type[n];
            };
            
            template <typename T, int>
            struct mean_type_impl
            {
                typedef T type;
            };
         
            template <typename T>
            struct mean_type_impl<T, 2>
            {
                typedef double type;
            };
         
            template <typename T>
            struct mean_type_impl<T, 3>
            {
				typedef typename boost::is_same<T, T>::type false_type;
                BOOST_STATIC_ASSERT_MSG(!false_type::value, "mean_type trait failed");
            };
        }

        template <typename value_type>
        struct mean_type
        {
            private:
                typedef value_type T;
                static T t;
                static detail::static_array<1> test(T);
                static detail::static_array<2> test(double);
                static detail::static_array<3> test(...);
            public:
                typedef typename detail::mean_type_impl<T, sizeof(test((t+t)/double(1)))/sizeof(char)>::type type;
        };

        template<>
        struct mean_type<double>
        {
            public:
                typedef double type;
        };
    }//end alea namespace 
}//end alps namespace
#endif // ALPS_NGS_ALEA_MEAN_TRAIT_HEADER
