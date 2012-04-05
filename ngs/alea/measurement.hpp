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


#ifndef ALPS_NGS_ALEA_MEASUREMENT_HEADER
#define ALPS_NGS_ALEA_MEASUREMENT_HEADER

#include <alps/ngs/alea/measurement_fwd.hpp>
#include <alps/ngs/alea/detail/accum_wrapper.hpp>
#include <alps/ngs/alea/extern_function.hpp>
#include <alps/ngs/stacktrace.hpp>

#include <boost/shared_ptr.hpp>


namespace alps
{
    namespace alea
    {
        //class that holds the base_wrapper pointer
        template<typename T> 
        measurement::measurement(T arg): base_(new detail::accumulator_wrapper<T>(arg)) 
        {}

        measurement::measurement(measurement const & arg): base_(arg.base_->clone()) 
        {}

        template<typename T>
        measurement& measurement::operator<<(const T& value) 
        {
            (*base_) << value; return *this;
        }

        template<typename T>
        detail::result_type_wrapper<T> &measurement::get() 
        {
            return (*base_).get<T>();
        }

        template <typename T>
        T& measurement::extract() 
        {
            return dynamic_cast<detail::accumulator_wrapper<T>&>(*base_).accum_;
        }
            
        std::ostream& operator<<(std::ostream &out, const measurement& m)
        {
            (*(m.base_)).print(out);
            return out;
        }
    }//end alea namespace 
}//end alps namespace
#endif // ALPS_NGS_ALEA_MEASUREMENT_HEADER
