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


#ifndef ALPS_NGS_ALEA_ACCUMULATOR_WRAPPER_HEADER
#define ALPS_NGS_ALEA_ACCUMULATOR_WRAPPER_HEADER

#include <alps/ngs/alea/detail/accum_wrapper.hpp>
#include <alps/ngs/alea/extern_function.hpp>
#include <alps/ngs/stacktrace.hpp>
#include <alps/ngs/alea/accumulator_wrapper_fwd.hpp>

#include <boost/shared_ptr.hpp>


namespace alps
{
    namespace alea
    {
        namespace detail
        {
            //class that holds the base_wrapper pointer
            template<typename T> 
            accumulator_wrapper::accumulator_wrapper(T arg): base_(new accumulator_wrapper_derived<T>(arg)) 
            {}

            template<typename T>
            accumulator_wrapper& accumulator_wrapper::operator<<(const T& value) 
            {
                (*base_) << value; return *this;
            }

            template<typename T>
            result_type_wrapper<T> &accumulator_wrapper::get() 
            {
                return (*base_).get<T>();
            }

            template <typename T>
            T & accumulator_wrapper::extract() 
            {
                return (dynamic_cast<detail::accumulator_wrapper_derived<T>& >(*base_)).accum_;
            }

            //TODO
            inline boost::uint64_t accumulator_wrapper::count() const
            {
                return (*base_).count();
            }
            
            accumulator_wrapper::accumulator_wrapper(accumulator_wrapper const & arg): base_(arg.base_->clone()) 
            {}
                
            std::ostream& operator<<(std::ostream &out, const accumulator_wrapper& m)
            {
                (*(m.base_)).print(out);
                return out;
            }
        }//end detail namespace 
    }//end alea namespace 
}//end alps namespace
#endif // ALPS_NGS_ALEA_ACCUMULATOR_WRAPPER_HEADER
