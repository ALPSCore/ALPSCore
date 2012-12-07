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

#ifndef ALPS_NGS_NUMERIC_VECTOR_HEADER
#define ALPS_NGS_NUMERIC_VECTOR_HEADER

#include <alps/ngs/stacktrace.hpp>

#include <boost/accumulators/numeric/functional/vector.hpp>
#include <boost/accumulators/numeric/functional.hpp>
#include <boost/throw_exception.hpp>
#include <boost/bind.hpp>

#include <vector>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace alps
{
    namespace ngs
    {
        namespace numeric
        {
            //------------------- operator += -------------------
            template<typename T>
            std::vector<T> & operator += (std::vector<T> & lhs, std::vector<T> const & rhs)
            {
                //------------------- init -------------------
                if(lhs.size() == 0)
                    lhs = std::vector<T>(rhs.size(), T());
                    
                if(lhs.size() != rhs.size())
                {
                    boost::throw_exception(std::runtime_error("vectors must have the same size!" + ALPS_STACKTRACE));
                    return lhs; //std::vector<T>() not possible bc ref
                }
                else
                {
                    using boost::numeric::operators::operator+=;
                    lhs += rhs;
                    return lhs;
                }
            }
            //------------------- operator + -------------------
            template<typename T>
            std::vector<T> operator + (std::vector<T> const & lhs, std::vector<T> const & rhs)
            {
                if(lhs.size() != rhs.size())
                {
                    boost::throw_exception(std::runtime_error("vectors must have the same size!" + ALPS_STACKTRACE));
                    return lhs;
                }
                else
                {
                    using boost::numeric::operators::operator+;
                    return lhs + rhs;
                }
            }
            //------------------- operator - -------------------
            template<typename T>
            std::vector<T> operator - (std::vector<T> const & lhs, std::vector<T> const & rhs)
            {
                if(lhs.size() != rhs.size())
                {
                    boost::throw_exception(std::runtime_error("vectors must have the same size!" + ALPS_STACKTRACE));
                    return std::vector<T>();
                }
                else
                {
                    using boost::numeric::operators::operator-;
                    return lhs - rhs;
                }
            }
            //------------------- operator * vector-vector-------------------
            template<typename T>
            std::vector<T> operator * (std::vector<T> const & lhs, std::vector<T> const & rhs)
            {
                if(lhs.size() != rhs.size())
                {
                    boost::throw_exception(std::runtime_error("vectors must have the same size!" + ALPS_STACKTRACE));
                    return lhs;
                }
                else
                {
                    using boost::numeric::operators::operator*;
                    return lhs*rhs;
                }
            }
            //------------------- operator / with scalar -------------------
            template<typename T, typename U>
            std::vector<T> operator / (std::vector<T> const & lhs, U const & scalar)
            {
                using boost::numeric::operators::operator/;
                return lhs / scalar;
            }
            //------------------- sqrt -------------------
            template<typename T> 
            std::vector<T> sqrt(std::vector<T> vec)
            {
                using std::sqrt;
                std::transform(vec.begin(), vec.end(), vec.begin(), static_cast<double (*)(double)>(&sqrt));
                return vec;
            }
            
        }//end namespace numeric
    }//end namespace ngs
}//end namespace alps


#endif //ALPS_NGS_NUMERIC_VECTOR_HEADER
