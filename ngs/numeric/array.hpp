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

#ifndef ALPS_NGS_NUMERIC_ARRAY_HEADER
#define ALPS_NGS_NUMERIC_ARRAY_HEADER

#include <alps/ngs/stacktrace.hpp>

#include <boost/throw_exception.hpp>
#include <boost/lambda/lambda.hpp>

#include <boost/array.hpp>

#include <algorithm>
#include <functional>
#include <cmath>
#include <stdexcept>

namespace alps
{
    namespace ngs //merged with alps/numerics/vector_function.hpp
    {
        namespace numeric
        {
            //------------------- operator += -------------------
            template<typename T, typename U, std::size_t N>
            boost::array<T, N> & operator += (boost::array<T, N> & lhs, boost::array<U, N> const & rhs)
            {
                if(lhs.size() != rhs.size())
                {
                    boost::throw_exception(std::runtime_error("arrays must have the same size!" + ALPS_STACKTRACE));
                    return lhs; //boost::array<T>() not possible bc ref
                }
                else
                {
                    std::transform(lhs.begin(), lhs.end(), rhs.begin(), lhs.begin(), std::plus<T>() );

                    return lhs;
                }
            }
            //------------------- operator + -------------------
            template<typename T, typename U, std::size_t N>
            boost::array<T, N> operator + (boost::array<T, N> lhs, boost::array<U, N> const & rhs)
            {
                if(lhs.size() != rhs.size())
                {
                    boost::throw_exception(std::runtime_error("arrays must have the same size!" + ALPS_STACKTRACE));
                    return lhs;
                }
                else
                {
                    std::transform(lhs.begin(), lhs.end(), rhs.begin(), lhs.begin(), std::plus<T>() );
                    
                    return lhs;
                }
            }
            //------------------- operator - -------------------
            template<typename T, typename U, std::size_t N>
            boost::array<T, N> operator - (boost::array<T, N> lhs, boost::array<U, N> const & rhs)
            {
                if(lhs.size() != rhs.size())
                {
                    boost::throw_exception(std::runtime_error("arrays must have the same size!" + ALPS_STACKTRACE));
                    return lhs;
                }
                else
                {
                    std::transform(lhs.begin(), lhs.end(), rhs.begin(), lhs.begin(), std::minus<T>() );
                    
                    return lhs;
                }
            }
            //------------------- operator * vector-vector-------------------
            template<typename T, typename U, std::size_t N>
            boost::array<T, N> operator * (boost::array<T, N> lhs, boost::array<U, N> const & rhs)
            {
                if(lhs.size() != rhs.size())
                {
                    boost::throw_exception(std::runtime_error("arrays must have the same size!" + ALPS_STACKTRACE));
                    return lhs;
                }
                else
                {
                    std::transform(lhs.begin(), lhs.end(), rhs.begin(), lhs.begin(), std::multiplies<T>());
                    
                    return lhs;
                }
            }
            //------------------- operator / with scalar -------------------
            //TODO: remove fix. Had a problem with the lambda and mac clang...
            template<typename T, typename U>
            class quick_fix_devide
            {
            public:
                quick_fix_devide(U const sca): sca_(sca) {}
                T operator()(T in) {return in/sca_;}
                U sca_;
            };
        
            template<typename T, std::size_t N, typename U>
            boost::array<T, N> operator / (boost::array<T, N> lhs, U const & scalar)
            {
                //using boost::lambda::_1;
                std::transform(lhs.begin(), lhs.end(), lhs.begin(), quick_fix_devide<T,U>(scalar));
                
                return lhs;
            }
            //------------------- sqrt -------------------
            template<typename T, std::size_t N> 
            boost::array<T, N> sqrt(boost::array<T, N> lhs) //sink argument
            {
                using std::sqrt;
                
                std::transform(lhs.begin(), lhs.end(), lhs.begin(), static_cast<double (*)(double)>(&sqrt));
                
                return lhs;
            }
            
        }//end namespace numeric
    }//end namespace ngs
}//end namespace alps

#endif //ALPS_NGS_NUMERIC_ARRAY_HEADER
