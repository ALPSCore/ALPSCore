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
#ifndef ALPS_NGS_NUMERIC_MULTI_ARRAY_HEADER
#define ALPS_NGS_NUMERIC_MULTI_ARRAY_HEADER

#include <alps/ngs/stacktrace.hpp>
#include <alps/ngs/short_print.hpp>

#include <boost/throw_exception.hpp>
#include <boost/lambda/lambda.hpp>

#include <boost/array.hpp>
#include <boost/multi_array.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace alps
{
    namespace ngs
    {
        namespace numeric
        {
            //------------------- operator += -------------------
            template<typename T, std::size_t N>
            boost::multi_array<T, N> & operator += (boost::multi_array<T, N> & lhs, boost::multi_array<T, N> const & rhs)
            {
                if(lhs.num_elements() == 0)
                {
                    lhs = rhs;
                    boost::array<std::size_t, N> shp;
                    std::copy(rhs.shape(), rhs.shape() + N, shp.begin());
                    lhs.resize(shp);
                }
                
                if(lhs.size() != rhs.size())
                {
                    boost::throw_exception(std::runtime_error("arrays must have the same size!" + ALPS_STACKTRACE));
                    return lhs; //boost::multi_array<T>() not possible bc ref
                }
                else
                {
                    for(int i = 0; i < lhs.num_elements(); ++i)
                    {
                        *(lhs.origin()+i) += *(rhs.origin()+i);
                    }

                    return lhs;
                }
            }
            //------------------- operator + -------------------
            template<typename T, std::size_t N>
            boost::multi_array<T, N> operator + (boost::multi_array<T, N> lhs, boost::multi_array<T, N> const & rhs)
            {
                if(lhs.size() != rhs.size())
                {
                    //TODO: fix small error created by example_accumulator_set
                    boost::throw_exception(std::runtime_error("arrays must have the same size!" + ALPS_STACKTRACE));
                    return lhs;
                }
                else
                {
                    for(int i = 0; i < lhs.num_elements(); ++i)
                    {
                        *(lhs.origin()+i) += *(rhs.origin()+i);
                    }
                    
                    return lhs;
                }
            }
            //------------------- operator - -------------------
            template<typename T, std::size_t N>
            boost::multi_array<T, N> operator - (boost::multi_array<T, N> lhs, boost::multi_array<T, N> const & rhs)
            {
                if(lhs.size() != rhs.size())
                {
                    boost::throw_exception(std::runtime_error("arrays must have the same size!" + ALPS_STACKTRACE));
                    return lhs;
                }
                else
                {
                    for(int i = 0; i < lhs.num_elements(); ++i)
                    {
                        *(lhs.origin()+i) -= *(rhs.origin()+i);
                    }
                    
                    return lhs;
                }
            }
            //~ //------------------- operator * vector-vector-------------------
            template<typename T, std::size_t N>
            boost::multi_array<T, N> operator * (boost::multi_array<T, N> lhs, boost::multi_array<T, N> const & rhs)
            {
                if(lhs.size() != rhs.size())
                {
                    boost::throw_exception(std::runtime_error("arrays must have the same size!" + ALPS_STACKTRACE));
                    return lhs;
                }
                else
                {
                    for(int i = 0; i < lhs.num_elements(); ++i)
                    {
                        *(lhs.origin()+i) *= *(rhs.origin()+i);
                    }
                    
                    return lhs;
                }
            }
            //------------------- operator / with scalar -------------------
            template<typename T, std::size_t N, typename U>
            boost::multi_array<T, N> operator / (boost::multi_array<T, N> lhs, U const & scalar)
            {
                for(int i = 0; i < lhs.num_elements(); ++i)
                {
                    *(lhs.origin()+i) /= scalar;
                }
                
                return lhs;
            }
            //------------------- sqrt -------------------
            template<typename T, std::size_t N> 
            boost::multi_array<T, N> sqrt(boost::multi_array<T, N> lhs) //sink argument
            {
                using std::sqrt;
                
                for(int i = 0; i < lhs.num_elements(); ++i)
                {
                    *(lhs.origin()+i) = sqrt(*(lhs.origin()+i));
                }

                
                return lhs;
            }
            
        }//end namespace numeric
    }//end namespace ngs
}//end namespace alps


#endif //ALPS_NGS_NUMERIC_MULTI_ARRAY_HEADER
