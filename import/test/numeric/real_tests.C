/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2012        by Michele Dolfi <dolfim@phys.ethz.ch>                *
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

#include <iostream>
#include <iterator>
#include <complex>
#include <vector>
#include <alps/numeric/real.hpp>

#define BOOST_TEST_SOURCE
#define BOOST_TEST_MODULE alps::numeric::real
#ifndef ALPS_LINK_BOOST_TEST
#include <boost/test/included/unit_test.hpp>
#else
#include <boost/test/unit_test.hpp>
#endif
#include <boost/mpl/list.hpp>

//
// List of types T for which the real(T) is tested
//
typedef boost::mpl::list<
      float
    , double
    , std::complex<float>
    , std::complex<double>
    , std::vector<float>
    , std::vector<double>
    , std::vector<std::complex<float> >
    , std::vector<std::complex<double> >
    , std::vector<std::vector<float> >
    , std::vector<std::vector<double> >
    , std::vector<std::vector<std::complex<float> > >
    , std::vector<std::vector<std::complex<double> > >
> test_types;


//
// ostream overloads
//
// template <typename T>
// std::ostream& operator<< (std::ostream& os, std::vector<T const&> vec)
// {
//     os << "[";
//     std::copy(vec.begin(), vec.end(), std::ostream_iterator<T const&>(os," "));
//     os << "]";
//     return os;
// }
template <typename T>
std::ostream& operator<< (std::ostream& os, std::vector<T> const & vec)
{
    os << "[";
    for (std::size_t i=0; i<vec.size(); ++i)
        os << (i ? " " : "") << vec[i];
    os << "]";
    return os;
}


//
// Filling functions
//
float fill_val = 1;
const std::size_t vecsize = 2;
template <typename T>
void fill (T & v)
{
    v = (fill_val++);
}
template <typename T>
void fill (std::complex<T> & v) {
    T real_part = fill_val++;
    T imag_part = fill_val++;
    v = std::complex<T>(real_part,imag_part);
}
template <typename T>
void fill (std::vector<T> & v)
{
    v.resize(vecsize);
    std::for_each(v.begin(), v.end(), static_cast<void (*)(T &)>(&fill));
}



//
// Test with full namespaces
//
BOOST_AUTO_TEST_CASE_TEMPLATE( real_with_namespace, T, test_types )
{
    T val; fill(val);
    std::cout << "real( " << val << " ) = " << alps::numeric::real(val) << std::endl;
}

//
// Test letting the compiler resolve the overloads
//
BOOST_AUTO_TEST_CASE_TEMPLATE( real_without_namespace, T, test_types )
{
    T val; fill(val);
    using alps::numeric::real;
    std::cout << "real( " << val << " ) = " << real(val) << std::endl;
}

