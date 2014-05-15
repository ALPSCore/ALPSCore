/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1999-2012 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
*                            Synge Todo <wistaria@comp-phys.org>,
*                            Andreas Hehn <hehn@phys.ethz.ch>
*
* This software is part of the ALPS libraries, published under the ALPS
* Library License; you can use, redistribute it and/or modify it under
* the terms of the license, either version 1 or (at your option) any later
* version.
* 
* You should have received a copy of the ALPS Library License along with
* the ALPS Libraries; see the file LICENSE.txt. If not, the license is also
* available from http://alps.comp-phys.org/.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
* FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT 
* SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE 
* FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE, 
* ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
* DEALINGS IN THE SOFTWARE.
*
*****************************************************************************/

/* $Id$ */

#ifndef ALPS_NUMERIC_REAL_HPP
#define ALPS_NUMERIC_REAL_HPP

#include <boost/type_traits/is_fundamental.hpp>
#include <boost/static_assert.hpp>
#include <algorithm>
#include <complex>
#include <vector>

namespace alps { namespace numeric {

// *********************************************************** REAL_TYPE TRAITS
template <class T>
struct real_type
{
    BOOST_STATIC_ASSERT((boost::is_fundamental<T>::value));
    typedef T type;
};

template <class T>
struct real_type<std::complex<T> >
{
    typedef T type;
};

template <class T>
struct real_type<std::vector<T> >
{
    typedef std::vector<typename real_type<T>::type> type;
};



// ************************************************************** REAL FUNCTION

template <class T> inline typename real_type<T>::type real(T );

namespace detail {
    
    // generic hook
    template <typename T>
    struct real_hook;

    // generic impl
    template <typename T, bool>
    struct real_impl;

    // overload for fundamental
    template <typename T>
    struct real_impl<T, true>{
        static inline T apply(T x)
        {
            return x;
        }
    };

    // forward to hooks
    template <typename T>
    struct real_impl<T, false>{
        static inline typename real_type<T>::type apply(T const& x)
        {
            return real_hook<T>::apply(x);
        }
    };



    // overload for complex of fundamental
    template <typename T>
    struct real_hook<std::complex<T> >{
        static inline T apply(std::complex<T> const& x)
        {
            BOOST_STATIC_ASSERT((boost::is_fundamental<T>::value));
            return std::real(x);
        }
    };

    // overload for std::vector
    template <typename T>
    struct real_hook<std::vector<T> >{
        static inline std::vector<T> const& apply(std::vector<T> const& x)
        {
            BOOST_STATIC_ASSERT((boost::is_fundamental<T>::value));
            return x;
        }
    };
    template <typename T>
    struct real_hook<std::vector<std::complex<T> > >{
        static inline std::vector<T> apply(std::vector<std::complex<T> > const& x)
        {
              std::vector<T> re;
              re.reserve(x.size());
              std::transform(x.begin(),x.end(),std::back_inserter(re),
                             static_cast<T (*)(std::complex<T> const &)>(&real_hook<std::complex<T> >::apply));
              return re;
        }
    };
    
    // overload for std::vector<std::vector>
    template <typename T>
    struct real_hook<std::vector<std::vector<T> > >{
        static inline std::vector<std::vector<T> > const& apply(std::vector<std::vector<T> > const& x)
        {
            BOOST_STATIC_ASSERT((boost::is_fundamental<T>::value));
            return x;
        }
    };
    template <typename T>
    struct real_hook<std::vector<std::vector<std::complex<T> > > >{
        static inline std::vector<std::vector<T> > apply(std::vector<std::vector<std::complex<T> > > const& x)
        {
            std::vector<std::vector<T> > re;
            re.reserve(x.size());
            std::transform(x.begin(),x.end(),std::back_inserter(re),
                           static_cast<std::vector<T> (*)(std::vector<std::complex<T> > const &)>(&real_hook<std::vector<std::complex<T> > >::apply));
            return re;
        }
    };
    
} // end namespace detail


template <class T>
inline typename real_type<T>::type real(T x)
{
    return detail::real_impl<T, boost::is_fundamental<T>::value>::apply(x);
}



} } // end namespace alps::numeric

#endif // ALPS_MATH_HPP
