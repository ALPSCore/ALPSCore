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


#ifndef ALPS_NGS_ALEA_EXTERN_FUNCTION_HEADER
#define ALPS_NGS_ALEA_EXTERN_FUNCTION_HEADER

#include <alps/ngs/alea/wrapper/accumulator_wrapper_fwd.hpp>
#include <alps/ngs/alea/global_enum.hpp>

namespace alps
{
    namespace alea
    {
        //------------------- for accumulator_wrapper -------------------
        template <typename Accum>
        Accum & extract(detail::accumulator_wrapper &m)
        {
            return m.extract<Accum>();
        } 

        template <typename Accum>
        inline boost::uint64_t count(Accum const & arg)
        {
            return arg.count();
        }

        namespace detail
        {
            //this one is needed, bc of name-collision in accum_wrapper
            template<typename Accum>
            boost::uint64_t count_wrap(Accum const & arg)
            {
                return count(arg);
            }
        }
    }//end alea namespace 
}//end alps namespace
#endif //ALPS_NGS_ALEA_EXTERN_FUNCTION_HEADER
