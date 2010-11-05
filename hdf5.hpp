// Copyright (C) 2008 - 2010 Lukas Gamper <gamperl -at- gmail.com>
// Use, modification and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#ifndef ALPS_HDF5_HPP
#define ALPS_HDF5_HPP

#ifndef _HDF5USEDLL_
# define _HDF5USEDLL_
#endif
#ifndef _HDF5USEHLDLL_
# define _HDF5USEHLDLL_
#endif

#include <map>
#include <set>
#include <list>
#include <deque>
#include <vector>
#include <string>
#include <complex>
#include <sstream>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <valarray>
#include <typeinfo>
#include <iostream>
#include <stdexcept>

#include <boost/ref.hpp>
#include <boost/any.hpp>
#include <boost/array.hpp>
#include <boost/config.hpp>
#include <boost/mpl/or.hpp>
#include <boost/mpl/if.hpp>
#include <boost/utility.hpp>
#include <boost/mpl/and.hpp>
#include <boost/integer.hpp>
#include <boost/optional.hpp>
#include <boost/weak_ptr.hpp>
#include <boost/filesystem.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/multi_array.hpp>
#include <boost/type_traits.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/shared_array.hpp>
#include <boost/static_assert.hpp>
#include <boost/intrusive_ptr.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/bindings/ublas.hpp>
#include <boost/type_traits/add_reference.hpp>
#include <boost/date_time/gregorian/gregorian.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include <hdf5.h>

namespace alps {
    #ifndef ALPS_HDF5_SZIP_BLOCK_SIZE
        #define ALPS_HDF5_SZIP_BLOCK_SIZE 32
    #endif

    namespace hdf5 {
        #define ALPS_HDF5_STRINGIFY(arg) ALPS_HDF5_STRINGIFY_HELPER(arg)
        #define ALPS_HDF5_STRINGIFY_HELPER(arg) #arg
        #define ALPS_HDF5_THROW_ERROR(error, message)                                                                                                      \
            {                                                                                                                                              \
                std::ostringstream buffer;                                                                                                                 \
                buffer << "Error in " << __FILE__ << " on " << ALPS_HDF5_STRINGIFY(__LINE__) << " in " << __FUNCTION__ << ":" << std::endl << message;     \
                throw ( error (buffer.str()));                                                                                                             \
            }
        #define ALPS_HDF5_THROW_RUNTIME_ERROR(message)                                                                                                     \
            ALPS_HDF5_THROW_ERROR(std::runtime_error, message)
        #define ALPS_HDF5_THROW_RANGE_ERROR(message)                                                                                                       \
            ALPS_HDF5_THROW_ERROR(std::range_error, message)

        namespace detail {
            namespace internal_state_type {
                typedef enum { CREATE, PLACEHOLDER } type;
            }
            struct internal_log_type {
                char * time;
                char * name;
            };
            struct internal_complex_type {
               double r;
               double i;
            };
        }

        template<typename T> std::vector<hsize_t> call_get_extent(T const &);
        template<typename T> void call_set_extent(T &, std::vector<std::size_t> const &);
        template<typename T> std::vector<hsize_t> call_get_offset(T const &);
        template<typename T> bool call_is_vectorizable(T const &);
        template<typename T, typename U> U const * call_get_data(
              std::vector<U> &
            , T const &
            , std::vector<hsize_t> const &
            , boost::optional<std::vector<hsize_t> > const & t = boost::none_t()
        );
        template<typename T, typename U> void call_set_data(T &, std::vector<U> const &, std::vector<hsize_t> const &, std::vector<hsize_t> const &);

        template<typename T> struct serializable_type { typedef T type; };
        template<typename T> struct native_type { typedef T type; };
        template<typename T> struct is_native : public boost::mpl::and_<
            typename boost::is_scalar<T>::type,
            typename boost::mpl::not_<typename boost::is_enum<T>::type>::type
        >::type {};
        template<typename T> std::vector<hsize_t> get_extent(T const &) { return std::vector<hsize_t>(1, 1); }
        template<typename T> void set_extent(T &, std::vector<std::size_t> const &) {}
        template<typename T> std::vector<hsize_t> get_offset(T const &) { return std::vector<hsize_t>(1, 1); }
        template<typename T> bool is_vectorizable(T const &) { 
            return boost::is_same<T, std::string>::value || boost::is_same<T, detail::internal_state_type::type>::value || (boost::is_scalar<T>::value && !boost::is_enum<T>::value); }
        template<typename T> typename boost::disable_if<typename boost::mpl::or_<
            typename boost::is_same<T, std::string>::type, typename boost::is_same<T, detail::internal_state_type::type>::type
        >::type, typename serializable_type<T>::type const *>::type get_data(
              std::vector<typename serializable_type<T>::type> & m
            , T const & v
            , std::vector<hsize_t> const &
            , std::vector<hsize_t> const & = std::vector<hsize_t>()
        ) {
            return &v;
        }
        namespace detail {
            template<typename T, typename U> void set_data_impl(T & v, std::vector<U> const & u, std::vector<hsize_t> const & s, std::vector<hsize_t> const & c, boost::mpl::false_) {
                if (s.size() != 1 || c.size() != 1 || c[0] == 0 || u.size() < c[0])
                    ALPS_HDF5_THROW_RANGE_ERROR("invalid data size")
                std::copy(u.begin(), u.begin() + c[0], &v + s[0]);
            }
            template<typename T, typename U> void set_data_impl(T &, std::vector<U> const &, std::vector<hsize_t> const &, std::vector<hsize_t> const &, boost::mpl::true_) { 
                ALPS_HDF5_THROW_RUNTIME_ERROR("invalid type conversion")
            }
        }
        template<typename T, typename U> void set_data(T & v, std::vector<U> const & u, std::vector<hsize_t> const & s, std::vector<hsize_t> const & c) {
            detail::set_data_impl(v, u, s, c, typename boost::mpl::or_<
                  typename boost::is_same<U, std::complex<double> >::type
                , typename boost::is_same<U, char *>::type
                , typename boost::is_enum<T>::type
                , typename boost::mpl::not_<typename boost::is_scalar<T>::type>::type
            >::type());
        }

        template<> struct serializable_type<bool> { typedef boost::int8_t type; };
        template<> struct native_type<bool> { typedef boost::int8_t type; };
        template<> struct is_native<bool> : public boost::mpl::true_ {};
        template<> struct is_native<bool const> : public boost::mpl::true_ {};
        template<typename T> typename boost::enable_if<
            typename boost::is_same<T, bool>::type, typename serializable_type<T>::type const *
        >::type get_data(
              std::vector<serializable_type<bool>::type> & m
            , T const & v
            , std::vector<hsize_t> const &
            , std::vector<hsize_t> const & = std::vector<hsize_t>()
        ) {
            m.resize(1);
            return &(m[0] = v);
        }

        template<> struct serializable_type<std::string> { typedef char const * type; };
        template<> struct native_type<std::string> { typedef std::string type; };
        template<> struct is_native<std::string> : public boost::mpl::true_ {};
        template<> struct is_native<std::string const> : public boost::mpl::true_ {};
        template<typename T> typename boost::enable_if<
            typename boost::is_same<T, std::string>::type, typename serializable_type<T>::type const *
        >::type get_data(
              std::vector<serializable_type<std::string>::type> & m
            , T const & v
            , std::vector<hsize_t> const &
            , std::vector<hsize_t> const & = std::vector<hsize_t>()
        ) {
            m.resize(1);
            return &(m[0] = v.c_str());
        }
        template<typename T> typename boost::disable_if<typename boost::mpl::or_<
            typename boost::is_same<T, std::complex<double> >::type,
            typename boost::is_same<T, char *>::type
        >::type>::type set_data(std::string & v, std::vector<T> const & u, std::vector<hsize_t> const & s, std::vector<hsize_t> const & c) { 
            if (s.size() != 1 || c.size() != 1 || c[0] == 0 || u.size() < c[0])
                ALPS_HDF5_THROW_RANGE_ERROR("invalid data size")
            for (std::string * w = &v + s[0]; w != &v + s[0] + c[0]; ++w)
                *w = boost::lexical_cast<std::string>(u[s[0] + (w - &v)]);
        }
        template<typename T> typename boost::enable_if<
            typename boost::is_same<T, char *>::type
        >::type set_data(std::string & v, std::vector<T> const & u, std::vector<hsize_t> const & s, std::vector<hsize_t> const & c) { 
            if (s.size() != 1 || c.size() != 1 || c[0] == 0 || u.size() < c[0])
                ALPS_HDF5_THROW_RANGE_ERROR("invalid data size")
            std::copy(u.begin(), u.begin() + c[0], &v);
        }
        template<typename T> typename boost::enable_if<
            typename boost::is_same<T, std::complex<double> >::type
        >::type set_data(std::string &, std::vector<T> const &, std::vector<hsize_t> const &, std::vector<hsize_t> const &) { 
            ALPS_HDF5_THROW_RUNTIME_ERROR("invalid type conversion")
        }

        template<> struct serializable_type<detail::internal_state_type::type> { typedef detail::internal_state_type::type type; };
        template<> struct native_type<detail::internal_state_type::type> { typedef detail::internal_state_type::type type; };
        template<> struct is_native<detail::internal_state_type::type> : public boost::mpl::true_ {};
        template<> struct is_native<detail::internal_state_type::type const> : public boost::mpl::true_ {};
        template<typename T> typename boost::enable_if<
            typename boost::is_same<T, detail::internal_state_type::type>::type, typename serializable_type<T>::type const *
        >::type get_data(
              std::vector<serializable_type<detail::internal_state_type::type>::type> & m
            , T const & v
            , std::vector<hsize_t> const &
            , std::vector<hsize_t> const & = std::vector<hsize_t>()
        ) {
            m.resize(1);
            return &(m[0] = v);
        }
        template<typename T> void set_data(detail::internal_state_type::type & v, std::vector<T> const & u, std::vector<hsize_t> const & s, std::vector<hsize_t> const &) {
            v = u.front();
        }

        template<typename T> bool is_vectorizable(std::complex<T> const &) { return true; }
        #ifdef ALPS_HDF5_WRITE_PYTHON_COMPATIBLE_COMPLEX
            template<typename T> struct serializable_type<std::complex<T> > { typedef detail::internal_complex_type type; };
            template<typename T> struct native_type<std::complex<T> > { typedef std::complex<T> type; };
            template<typename T> struct is_native<std::complex<T> > : public boost::mpl::true_ {};
            template<typename T> struct is_native<std::complex<T> const > : public boost::mpl::true_ {};
            template<typename T> std::vector<hsize_t> get_offset(std::complex<T> const &) { return std::vector<hsize_t>(1, 1); }
            template<typename T> void set_extent(std::complex<T>  &, std::vector<std::size_t> const & s) {
                if (!is_native<T>::value)
                    ALPS_HDF5_THROW_RUNTIME_ERROR("complex can only be build over scalar data types")
                if(s.size() > 0)
                    ALPS_HDF5_THROW_RANGE_ERROR("invalid data size")
            }
            template<typename T> std::vector<hsize_t> get_extent(std::complex<T> const &) { return std::vector<hsize_t>(1, 1); }
            template<typename T> typename serializable_type<std::complex<T> >::type const * get_data(
                  std::vector<typename serializable_type<std::complex<T> >::type> & m
                , std::complex<T> const & v
                , std::vector<hsize_t> const &
                , std::vector<hsize_t> const & t = std::vector<hsize_t>(1, 1)
            ) {
                if (t.size() != 1 || t[0] == 0)
                    ALPS_HDF5_THROW_RANGE_ERROR("invalid data size")
                m.resize(t[0]);
                for (std::complex<T> const * u = &v; u != &v + t[0]; ++u) {
                    detail::internal_complex_type c = { u->real(), u->imag() };
                    m[u - &v] = c;
                }
                return &m[0];
            }
        #else
            template<typename T> struct serializable_type<std::complex<T> > {
                typedef typename serializable_type<typename boost::remove_const<T>::type>::type type;
            };
            template<typename T> struct native_type<std::complex<T> > {
                typedef typename native_type<typename boost::remove_const<T>::type>::type type;
            };
            template<typename T> struct is_native<std::complex<T> > : public boost::mpl::false_ {};
            template<typename T> struct is_native<std::complex<T> const > : public boost::mpl::false_ {};
            template<typename T> std::vector<hsize_t> get_offset(std::complex<T> const &) { return std::vector<hsize_t>(1, 2); }
            template<typename T> void set_extent(std::complex<T>  &, std::vector<std::size_t> const & s) {
                if (!is_native<T>::value)
                    ALPS_HDF5_THROW_RUNTIME_ERROR("complex can only be build over scalar data types")
                if(s.size() != 1)
                    ALPS_HDF5_THROW_RANGE_ERROR("invalid data size")
            }
            template<typename T> std::vector<hsize_t> get_extent(std::complex<T> const & v) {
                if (!is_native<T>::value)
                    ALPS_HDF5_THROW_RUNTIME_ERROR("complex can only be build over scalar data types")
                return std::vector<hsize_t>(1, 2);
            }
            template<typename T> typename serializable_type<std::complex<T> >::type const * get_data(
                  std::vector<typename serializable_type<std::complex<T> >::type> & m
                , std::complex<T> const & v
                , std::vector<hsize_t> const & s
                , std::vector<hsize_t> const & = std::vector<hsize_t>()
            ) {
                return reinterpret_cast<typename serializable_type<std::complex<T> >::type const *> (&v);
            }
        #endif
        template<typename T, typename U> typename boost::disable_if<typename boost::mpl::or_<
            typename boost::is_same<U, T>::type,
            typename boost::is_same<U, std::complex<T> >::type
        >::type>::type set_data(std::complex<T> &, std::vector<U> const &, std::vector<hsize_t> const &, std::vector<hsize_t> const &) {
            ALPS_HDF5_THROW_RUNTIME_ERROR("invalid type conversion")
        }
        template<typename T> void set_data(std::complex<T> & v, std::vector<T> const & u, std::vector<hsize_t> const & s, std::vector<hsize_t> const & c) {
            if (s.size() != 1 || c.size() != 1 || c[0] == 0 || u.size() < c[0])
                ALPS_HDF5_THROW_RANGE_ERROR("invalid data size")
            std::copy(u.begin(), u.begin() + c[0], reinterpret_cast<T *>(&v) + s[0]);
        }
        template<typename T> void set_data(std::complex<T> & v, std::vector<std::complex<double> > const & u, std::vector<hsize_t> const & s, std::vector<hsize_t> const & c) {
            if (s.size() != 1 || c.size() != 1 || c[0] == 0 || u.size() < c[0])
                ALPS_HDF5_THROW_RANGE_ERROR("invalid data size")
            std::copy(u.begin(), u.begin() + c[0], &v + s[0]);
        }

        #define HDF5_DEFINE_VECTOR_TYPE(C)                                                                                                                 \
            template<typename T> struct serializable_type< C <T> > {                                                                                       \
                typedef typename serializable_type<typename boost::remove_const<T>::type>::type type;                                                      \
            };                                                                                                                                             \
            template<typename T> struct native_type< C <T> > { typedef typename native_type<typename boost::remove_const<T>::type>::type type; };          \
            template<typename T> std::vector<hsize_t> get_extent( C <T> const & v) {                                                                       \
                std::vector<hsize_t> s(1, v.size());                                                                                                       \
                if (!is_native<T>::value && v.size()) {                                                                                                    \
                    std::vector<hsize_t> t(call_get_extent(v[0]));                                                                                         \
                    std::copy(t.begin(), t.end(), std::back_inserter(s));                                                                                  \
                    for (std::size_t i = 1; i < v.size(); ++i)                                                                                             \
                        if (!std::equal(t.begin(), t.end(), call_get_extent(v[i]).begin()))                                                                \
                            ALPS_HDF5_THROW_RANGE_ERROR("no rectengual matrix")                                                                            \
                }                                                                                                                                          \
                return s;                                                                                                                                  \
            }                                                                                                                                              \
            template<typename T> void set_extent( C <T> & v, std::vector<std::size_t> const & s) {                                                         \
                if(                                                                                                                                        \
                       !(s.size() == 1 && s[0] == 0)                                                                                                       \
                    && ((is_native<T>::value && s.size() != 1) || (!is_native<T>::value && s.size() < 2))                                                  \
                )                                                                                                                                          \
                    ALPS_HDF5_THROW_RANGE_ERROR("invalid data size")                                                                                       \
                v.resize(s[0]);                                                                                                                            \
                if (!is_native<T>::value)                                                                                                                  \
                    for (std::size_t i = 0; i < s[0]; ++i)                                                                                                 \
                        call_set_extent(v[i], std::vector<std::size_t>(s.begin() + 1, s.end()));                                                           \
            }                                                                                                                                              \
            template<typename T> std::vector<hsize_t> get_offset( C <T> const & v) {                                                                       \
                if (v.size() == 0)                                                                                                                         \
                    return std::vector<hsize_t>(1, 0);                                                                                                     \
                else if (is_native<T>::value && boost::is_same<typename native_type<T>::type, std::string>::value)                                         \
                    return std::vector<hsize_t>(1, 1);                                                                                                     \
                else if (is_native<T>::value)                                                                                                              \
                    return call_get_extent(v);                                                                                                             \
                else {                                                                                                                                     \
                   std::vector<hsize_t> c(1, 1), d(call_get_offset(v[0]));                                                                                 \
                    std::copy(d.begin(), d.end(), std::back_inserter(c));                                                                                  \
                    return c;                                                                                                                              \
                }                                                                                                                                          \
            }                                                                                                                                              \
            template<typename T> bool is_vectorizable( C <T> const & v) {                                                                                  \
                for(std::size_t i = 0; i < v.size(); ++i)                                                                                                  \
                    if (!call_is_vectorizable(v[i]) || call_get_extent(v[0])[0] != call_get_extent(v[i])[0])                                               \
                        return false;                                                                                                                      \
                return true;                                                                                                                               \
            }                                                                                                                                              \
            template<typename T> typename serializable_type< C <T> >::type const * get_data(                                                               \
                  std::vector<typename serializable_type< C <T> >::type> & m                                                                               \
                ,  C <T> const & v                                                                                                                         \
                , std::vector<hsize_t> const & s                                                                                                           \
                , std::vector<hsize_t> const & = std::vector<hsize_t>()                                                                                    \
            ) {                                                                                                                                            \
                if (is_native<T>::value)                                                                                                                   \
                    return call_get_data(m, (const_cast< C <T> &>(v))[s[0]], std::vector<hsize_t>(s.begin() + 1, s.end()), call_get_extent(v));            \
                else                                                                                                                                       \
                    return call_get_data(m, (const_cast< C <T> &>(v))[s[0]], std::vector<hsize_t>(s.begin() + 1, s.end()));                                \
            }                                                                                                                                              \
            template<typename T, typename U> void set_data(                                                                                                \
                C <T> & v, std::vector<U> const & u, std::vector<hsize_t> const & s, std::vector<hsize_t> const & c                                        \
            ) {                                                                                                                                            \
                if (is_native<T>::value)                                                                                                                   \
                    call_set_data(v[s[0]], u, s, c);                                                                                                       \
                else                                                                                                                                       \
                    call_set_data(v[s[0]], u, std::vector<hsize_t>(s.begin() + 1, s.end()), std::vector<hsize_t>(c.begin() + 1, c.end()));                 \
            }
        HDF5_DEFINE_VECTOR_TYPE(std::vector)
        HDF5_DEFINE_VECTOR_TYPE(std::valarray)
        HDF5_DEFINE_VECTOR_TYPE(boost::numeric::ublas::vector)
        #undef HDF5_DEFINE_VECTOR_TYPE

        template<typename T> struct serializable_type<std::pair<T *, std::vector<std::size_t> > > { typedef typename serializable_type<typename boost::remove_const<T>::type>::type type; };
        template<typename T> struct native_type<std::pair<T *, std::vector<std::size_t> > > { typedef typename native_type<typename boost::remove_const<T>::type>::type type; };
        template<typename T> std::vector<hsize_t> get_extent(std::pair<T *, std::vector<std::size_t> > const & v) {
            std::vector<hsize_t> s(v.second.begin(), v.second.end());
            if (!is_native<T>::value && v.second.size()) {
                 std::vector<hsize_t> t(call_get_extent(*v.first));
                 std::copy(t.begin(), t.end(), std::back_inserter(s));
                 for (std::size_t i = 1; i < std::accumulate(v.second.begin(), v.second.end(), std::size_t(1), std::multiplies<std::size_t>()); ++i)
                     if (!std::equal(t.begin(), t.end(), call_get_extent(*(v.first + i)).begin()))
                         ALPS_HDF5_THROW_RANGE_ERROR("no rectengual matrix")
            }
            return s;
        }
        template<typename T> void set_extent(std::pair<T *, std::vector<std::size_t> > & v, std::vector<std::size_t> const & s) {
            if (!(s.size() == 1 && s[0] == 0 && std::accumulate(v.second.begin(), v.second.end(), std::size_t(0)) == 0) && !std::equal(v.second.begin(), v.second.end(), s.begin()))
                ALPS_HDF5_THROW_RANGE_ERROR("invalid data size")
            if (s.size() == 1 && s[0] == 0)
                v.first = NULL;
            else if (!is_native<T>::value && s.size() > v.second.size())
                for (std::size_t i = 0; i < std::accumulate(v.second.begin(), v.second.end(), std::size_t(1), std::multiplies<hsize_t>()); ++i)
                    call_set_extent(*(v.first + i), std::vector<std::size_t>(s.begin() + v.second.size(), s.end()));
        }
        template<typename T> std::vector<hsize_t> get_offset(std::pair<T *, std::vector<std::size_t> > const & v) {
            if (is_native<T>::value && boost::is_same<typename native_type<std::pair<T *, std::vector<std::size_t> > >::type, std::string>::value)
                return std::vector<hsize_t>(v.second.size(), 1);
            else if (is_native<T>::value)
                return std::vector<hsize_t>(v.second.begin(), v.second.end());
            else {
                std::vector<hsize_t> c(v.second.size(), 1), d(call_get_offset(*v.first));
                std::copy(d.begin(), d.end(), std::back_inserter(c));
                return c;
            }
        }
        template<typename T> bool is_vectorizable(std::pair<T *, std::vector<std::size_t> > const & v) {
            for (std::size_t i = 0; i < std::accumulate(v.second.begin(), v.second.end(), std::size_t(1), std::multiplies<hsize_t>()); ++i)
                if (!call_is_vectorizable(v.first[i]) || call_get_extent(v.first[0])[0] != call_get_extent(v.first[i])[0])
                    return false;
            return true;
        }
        template<typename T> typename serializable_type<std::pair<T *, std::vector<std::size_t> > >::type const * get_data(
              std::vector<typename serializable_type<std::pair<T *, std::vector<std::size_t> > >::type> & m
            , std::pair<T *, std::vector<std::size_t> > const & v
            , std::vector<hsize_t> const & s
            , std::vector<hsize_t> const & = std::vector<hsize_t>()
        ) {
            hsize_t start = 0;
            for (std::size_t i = 0; i < v.second.size(); ++i)
                start += s[i] * std::accumulate(v.second.begin() + i + 1, v.second.end(), hsize_t(1), std::multiplies<hsize_t>());
            if (is_native<T>::value)
                return call_get_data(
                       m
                    , v.first[start]
                    , std::vector<hsize_t>(s.begin() + v.second.size(), s.end())
                    , std::vector<hsize_t>(1, std::accumulate(v.second.begin(), v.second.end(), hsize_t(1), std::multiplies<hsize_t>()))
                );
            else
                return call_get_data(
                       m
                     , v.first[start]
                     , std::vector<hsize_t>(s.begin() + v.second.size(), s.end())
                );
        }
        template<typename T, typename U> void set_data(std::pair<T *, std::vector<std::size_t> > & v, std::vector<U> const & u, std::vector<hsize_t> const & s, std::vector<hsize_t> const & c) {
            std::vector<hsize_t> start(1, 0);
            for (std::size_t i = 0; i < v.second.size(); ++i)
                start[0] += s[i] * std::accumulate(v.second.begin() + i + 1, v.second.end(), hsize_t(1), std::multiplies<hsize_t>());
            std::copy(s.begin() + v.second.size(), s.end(), std::back_inserter(start));
            if (is_native<T>::value)
                call_set_data(v.first[start[0]], u, start, std::vector<hsize_t>(1, std::accumulate(c.begin(), c.end(), hsize_t(1), std::multiplies<hsize_t>())));
            else
                call_set_data(v.first[start[0]], u, std::vector<hsize_t>(start.begin() + 1, start.end()), std::vector<hsize_t>(c.begin() + v.second.size(), c.end()));
        }

        namespace detail {
            class error {
                public:
                    static herr_t noop(hid_t) { return 0; }
                    static herr_t callback(unsigned n, H5E_error2_t const * desc, void * buffer) {
                        *reinterpret_cast<std::ostringstream *>(buffer) << "    #" << to_string(n) << " " << desc->file_name << " line " << to_string(desc->line) << " in " << desc->func_name << "(): " << desc->desc << std::endl;
                        return 0;
                    }
                    static std::string invoke(hid_t id) {
                        std::ostringstream buffer;
                        buffer << "HDF5 error: " << to_string(id) << std::endl;
                        H5Ewalk2(H5E_DEFAULT, H5E_WALK_DOWNWARD, callback, &buffer);
                        return buffer.str();
                    }
                private:
                    static std::string to_string(int arg) {
                        char buffer[255];
                        if (sprintf(buffer, "%i", arg) < 0)
                            ALPS_HDF5_THROW_RUNTIME_ERROR("error converting int to string");
                        return buffer;
                    }
            };
            template<herr_t(*F)(hid_t)> class resource {
                public:
                    resource(): _id(-1) {}
                    resource(hid_t id): _id(id) {
                        if (_id < 0)
                            ALPS_HDF5_THROW_RUNTIME_ERROR(error::invoke(_id))
                    }
                    ~resource() {
                        if(_id < 0 || (_id = F(_id)) < 0) {
                            std::cerr << "Error in " << __FILE__ << " on " << ALPS_HDF5_STRINGIFY(__LINE__) << " in " << __FUNCTION__ << ":" << std::endl << error::invoke(_id) << std::endl;
                            std::abort();
                        }
                    }
                    operator hid_t() const { 
                        return _id; 
                    }
                    resource<F> & operator=(hid_t id) { 
                        if ((_id = id) < 0) 
                            ALPS_HDF5_THROW_RUNTIME_ERROR(error::invoke(_id))
                        return *this; 
                    }
                private:
                    hid_t _id;
            };
            typedef resource<H5Fclose> file_type;
            typedef resource<H5Gclose> group_type;
            typedef resource<H5Dclose> data_type;
            typedef resource<H5Aclose> attribute_type;
            typedef resource<H5Sclose> space_type;
            typedef resource<H5Tclose> type_type;
            typedef resource<H5Pclose> property_type;
            typedef resource<error::noop> error_type;
            template <typename T> T check_file(T id) { file_type unused(id); return unused; }
            template <typename T> T check_group(T id) { group_type unused(id); return unused; }
            template <typename T> T check_data(T id) { data_type unused(id); return unused; }
            template <typename T> T check_attribute(T id) { attribute_type unused(id); return unused; }
            template <typename T> T check_space(T id) { space_type unused(id); return unused; }
            template <typename T> T check_type(T id) { type_type unused(id); return unused; }
            template <typename T> T check_property(T id) { property_type unused(id); return unused; }
            template <typename T> T check_error(T id) { error_type unused(id); return unused; }
            struct hdf5_context : boost::noncopyable {
                hdf5_context(std::string const & filename, hid_t file_id, bool compress)
                    : _compress(compress)
                    , _revision(0)
                    , _state_id(-1)
                    , _log_id(-1)
                    , _filename(filename)
                    , _file_id(file_id)
                {
                    if (_compress) {
                        unsigned int flag;
                        detail::check_error(H5Zget_filter_info(H5Z_FILTER_SZIP, &flag));
                        _compress = flag & H5Z_FILTER_CONFIG_ENCODE_ENABLED;
                    }
                }
                ~hdf5_context() {
                    try {
                        H5Fflush(_file_id, H5F_SCOPE_GLOBAL);
                        if (_state_id > -1)
                            detail::check_type(_state_id);
                        if (_log_id > -1)
                            detail::check_type(_log_id);
                        #ifndef ALPS_HDF5_CLOSE_GREEDY
                            if (
                                H5Fget_obj_count(_file_id, H5F_OBJ_DATATYPE) > (_state_id == -1 ? 0 : 1) + (_log_id == -1 ? 0 : 1)
                                || H5Fget_obj_count(_file_id, H5F_OBJ_ALL) - H5Fget_obj_count(_file_id, H5F_OBJ_FILE) - H5Fget_obj_count(_file_id, H5F_OBJ_DATATYPE) > 0
                            ) {
                                std::cerr << "Not all resources closed in file '" << _filename << "'" << std::endl;
                                std::abort();
                            }
                        #endif
                    } catch (std::exception & ex) {
                        std::cerr << "Error destructing HDF5 context of file '" << _filename << "'\n" << ex.what() << std::endl;
                        std::abort();
                    }
                }
                bool _compress;
                // TODO: this has to be checkd before every operation to make sure its still the same
                int _revision;
                hid_t _state_id;
                hid_t _log_id;
                hid_t _complex_id;
                std::string _filename;
                file_type _file_id;
            };
            #define HDF5_FOREACH_SCALAR(callback)                                                                                                          \
                callback(char)                                                                                                                             \
                callback(signed char)                                                                                                                      \
                callback(unsigned char)                                                                                                                    \
                callback(short)                                                                                                                            \
                callback(unsigned short)                                                                                                                   \
                callback(int)                                                                                                                              \
                callback(unsigned int)                                                                                                                     \
                callback(long)                                                                                                                             \
                callback(unsigned long)                                                                                                                    \
                callback(long long)                                                                                                                        \
                callback(unsigned long long)                                                                                                               \
                callback(float)                                                                                                                            \
                callback(double)                                                                                                                           \
                callback(long double)
        }

        template<typename T> std::vector<hsize_t> call_get_extent(T const & v) {
            return get_extent(v);
        }
        template<typename T> void call_set_extent(T & v, std::vector<std::size_t> const & s) {
            return set_extent(v, s);
        }
        template<typename T> std::vector<hsize_t> call_get_offset(T const & v) {
            return get_offset(v);
        }
        template<typename T> bool call_is_vectorizable(T const & v) {
            return is_vectorizable(v);
        }
        template<typename T, typename U> U const * call_get_data(
              std::vector<U> & m
            , T const & v
            , std::vector<hsize_t> const & s
            , boost::optional<std::vector<hsize_t> > const & c
        ) {
            return c ? get_data(m, v, s, *c) : get_data(m, v, s);
        }
        template<typename T, typename U> void call_set_data(T & v, std::vector<U> const & u, std::vector<hsize_t> const & s, std::vector<hsize_t> const & c) {
            return set_data(v, u, s, c);
        }

         template<hid_t(* F )(std::string const &)> class archive {
            public:
                struct log_type {
                    boost::posix_time::ptime time;
                    std::string name;
                };
                archive(archive<F> const & rhs)
                    : _path_context(rhs._path_context)
                    , _hdf5_context(rhs._hdf5_context)
                {}
                virtual ~archive() {
                    try {
                        H5Fflush(file_id(), H5F_SCOPE_GLOBAL);
                    } catch (std::exception & ex) {
                        std::cerr << "Error destructing archive of file '" << filename() << "'\n" << ex.what() << std::endl;
                        std::abort();
                    }
                }
                std::string const & filename() const {
                    return _hdf5_context->_filename;
                }
                std::string encode_segment(std::string const & s) {
                    std::string r = s;
                    char chars[] = {'&', '/'};
                    for (std::size_t i = 0; i < sizeof(chars); ++i)
                        for (std::size_t pos = r.find_first_of(chars[i]); pos < std::string::npos; pos = r.find_first_of(chars[i], pos + 1))
                            r = r.substr(0, pos) + "&#" + boost::lexical_cast<std::string, int>(chars[i]) + ";" + r.substr(pos + 1);
                    return r;
                }
                std::string decode_segment(std::string const & s) {
                    std::string r = s;
                    for (std::size_t pos = r.find_first_of('&'); pos < std::string::npos; pos = r.find_first_of('&', pos + 1))
                        r = r.substr(0, pos) + static_cast<char>(boost::lexical_cast<int>(r.substr(pos + 2, r.find_first_of(';', pos) - pos - 2))) + r.substr(r.find_first_of(';', pos) + 1);
                    return r;
                }
                void commit(std::string const & name = "") {
                    set_attr("/revisions/@last", ++_hdf5_context->_revision);
                    set_group("/revisions/" + boost::lexical_cast<std::string>(revision()));
                    std::string time = boost::posix_time::to_iso_string(boost::posix_time::second_clock::local_time());
                    detail::internal_log_type v = {
                        std::strcpy(new char[time.size() + 1], time.c_str()),
                        std::strcpy(new char[name.size() + 1], name.c_str())
                    };
                    set_attr("/revisions/" + boost::lexical_cast<std::string>(revision()) + "/@info", v);
                    delete[] v.time;
                    delete[] v.name;
                }
                std::vector<std::pair<std::string, std::string> > list_revisions() const {
                    // TODO: implement
                    return std::vector<std::pair<std::string, std::string> >();
                }
                void export_revision(std::size_t revision, std::string const & file) const {
                    // TODO: implement
                }
                std::string get_context() const {
                    return _path_context;
                }
                void set_context(std::string const & context) {
                    _path_context = context;
                }
                std::string complete_path(std::string const & p) const {
                    if (p.size() && p[0] == '/')
                        return p;
                    else if (p.size() < 2 || p.substr(0, 2) != "..")
                        return _path_context + (_path_context.size() == 1 || !p.size() ? "" : "/") + p;
                    else {
                        std::string s = _path_context;
                        std::size_t i = 0;
                        for (; s.size() && p.substr(i, 2) == ".."; i += 3)
                            s = s.substr(0, s.find_last_of('/'));
                        return s + (s.size() == 1 || !p.substr(i).size() ? "" : "/") + p.substr(i);
                    }
                }
                bool is_group(std::string const & p) const {
                    try {
                        hid_t id = H5Gopen2(file_id(), complete_path(p).c_str(), H5P_DEFAULT);
                        return id < 0 ? false : detail::check_group(id) != 0;
                    } catch (std::exception & ex) {
                        ALPS_HDF5_THROW_RUNTIME_ERROR("file: " + filename() + ", path: " + p + "\n" + ex.what());
                    }
                }
                bool is_data(std::string const & p) const {
                    try {
                        hid_t id = H5Dopen2(file_id(), complete_path(p).c_str(), H5P_DEFAULT);
                        return id < 0 ? false : detail::check_data(id) != 0;
                    } catch (std::exception & ex) {
                        ALPS_HDF5_THROW_RUNTIME_ERROR("file: " + filename() + ", path: " + p + "\n" + ex.what());
                    }
                }
                bool is_attribute(std::string const & p) const {
                    try {
                        if (p.find_last_of('@') == std::string::npos)
                            ALPS_HDF5_THROW_RUNTIME_ERROR("no attribute paht: " + complete_path(p))
                        hid_t parent_id;
                        if (is_group(complete_path(p).substr(0, complete_path(p).find_last_of('@') - 1)))
                            parent_id = detail::check_error(H5Gopen2(file_id(), complete_path(p).substr(0, complete_path(p).find_last_of('@') - 1).c_str(), H5P_DEFAULT));
                        else if (is_data(complete_path(p).substr(0, complete_path(p).find_last_of('@') - 1)))
                           parent_id = detail::check_error(H5Dopen2(file_id(), complete_path(p).substr(0, complete_path(p).find_last_of('@') - 1).c_str(), H5P_DEFAULT));
                        else
                            #ifdef ALPS_HDF5_READ_GREEDY
                                return false;
                            #else
                                ALPS_HDF5_THROW_RUNTIME_ERROR("unknown path: " + complete_path(p))
                            #endif
                        bool exists = detail::check_error(H5Aexists(parent_id, p.substr(p.find_last_of('@') + 1).c_str()));
                        if (is_group(complete_path(p).substr(0, complete_path(p).find_last_of('@') - 1)))
                            detail::check_group(parent_id);
                        else
                            detail::check_data(parent_id);
                        return exists;
                    } catch (std::exception & ex) {
                        ALPS_HDF5_THROW_RUNTIME_ERROR("file: " + filename() + ", path: " + p + "\n" + ex.what());
                    }
                }
                std::vector<std::size_t> extent(std::string const & p) const {
                    try {
                        if (is_null(p))
                            return std::vector<std::size_t>(1, 0);
                        else if (is_scalar(p))
                            return std::vector<std::size_t>(1, 1);
                        std::vector<hsize_t> buffer(dimensions(p), 0);
                        hid_t space_id;
                        if (p.find_last_of('@') != std::string::npos) {
                            detail::attribute_type attr_id(open_attribute(complete_path(p)));
                            space_id = H5Aget_space(attr_id);
                        } else {
                            detail::data_type data_id(H5Dopen2(file_id(), complete_path(p).c_str(), H5P_DEFAULT));
                            space_id = H5Dget_space(data_id);
                        }
                        detail::check_error(H5Sget_simple_extent_dims(space_id, &buffer.front(), NULL));
                        detail::check_space(space_id);
                        std::vector<std::size_t> extent(buffer.size(), 0);
                        std::copy(buffer.begin(), buffer.end(), extent.begin());
                        return extent;
                    } catch (std::exception & ex) {
                        ALPS_HDF5_THROW_RUNTIME_ERROR("file: " + filename() + ", path: " + p + "\n" + ex.what());
                    }
                }
                std::size_t dimensions(std::string const & p) const {
                    try {
                        if (p.find_last_of('@') != std::string::npos) {
                            detail::attribute_type attr_id(open_attribute(complete_path(p)));
                            return static_cast<hid_t>(detail::check_error(H5Sget_simple_extent_dims(detail::space_type(H5Aget_space(attr_id)), NULL, NULL)));
                        } else {
                            detail::data_type data_id(H5Dopen2(file_id(), complete_path(p).c_str(), H5P_DEFAULT));
                            return static_cast<hid_t>(detail::check_error(H5Sget_simple_extent_dims(detail::space_type(H5Dget_space(data_id)), NULL, NULL)));
                        }
                    } catch (std::exception & ex) {
                        ALPS_HDF5_THROW_RUNTIME_ERROR("file: " + filename() + ", path: " + p + "\n" + ex.what());
                    }
                }
                bool is_scalar(std::string const & p) const {
                    try {
                        hid_t space_id;
                        if (p.find_last_of('@') != std::string::npos) {
                            detail::attribute_type attr_id(open_attribute(complete_path(p)));
                            space_id = H5Aget_space(attr_id);
                        } else {
                            detail::data_type data_id(H5Dopen2(file_id(), complete_path(p).c_str(), H5P_DEFAULT));
                            space_id = H5Dget_space(data_id);
                        }
                        H5S_class_t type = H5Sget_simple_extent_type(space_id);
                        detail::check_space(space_id);
                        if (type == H5S_NO_CLASS)
                            ALPS_HDF5_THROW_RUNTIME_ERROR("error reading class " + complete_path(p))
                        return type == H5S_SCALAR;
                    } catch (std::exception & ex) {
                        ALPS_HDF5_THROW_RUNTIME_ERROR("file: " + filename() + ", path: " + p + "\n" + ex.what());
                    }
                }
                bool is_string(std::string const & p) const {
                    try {
                        hid_t type_id;
                        if (!is_scalar(p))
                            return false;
                        if (p.find_last_of('@') != std::string::npos) {
                            detail::attribute_type attr_id(open_attribute(complete_path(p)));
                            type_id = H5Aget_type(attr_id);
                        } else {
                            detail::data_type data_id(H5Dopen2(file_id(), complete_path(p).c_str(), H5P_DEFAULT));
                            type_id = H5Dget_type(data_id);
                        }
                        detail::type_type native_id(H5Tget_native_type(type_id, H5T_DIR_ASCEND));
                        detail::check_type(type_id);
                        return H5Tget_class(native_id) == H5T_STRING;
                    } catch (std::exception & ex) {
                        ALPS_HDF5_THROW_RUNTIME_ERROR("file: " + filename() + ", path: " + p + "\n" + ex.what());
                    }
                }
                bool is_null(std::string const & p) const {
                    try {
                        hid_t space_id;
                        if (p.find_last_of('@') != std::string::npos) {
                            detail::attribute_type attr_id(open_attribute(complete_path(p)));
                            space_id = H5Aget_space(attr_id);
                        } else {
                            detail::data_type data_id(H5Dopen2(file_id(), complete_path(p).c_str(), H5P_DEFAULT));
                            space_id = H5Dget_space(data_id);
                        }
                        H5S_class_t type = H5Sget_simple_extent_type(space_id);
                        detail::check_space(space_id);
                        if (type == H5S_NO_CLASS)
                            ALPS_HDF5_THROW_RUNTIME_ERROR("error reading class " + complete_path(p))
                        return type == H5S_NULL;
                    } catch (std::exception & ex) {
                        ALPS_HDF5_THROW_RUNTIME_ERROR("file: " + filename() + ", path: " + p + "\n" + ex.what());
                    }
                }
                void delete_data(std::string const & p) const {
                    try {
                        if (is_data(p))
                            // TODO: implement provenance
                            detail::check_error(H5Ldelete(file_id(), complete_path(p).c_str(), H5P_DEFAULT));
                        else
                            ALPS_HDF5_THROW_RUNTIME_ERROR("the path does not exists: " + p)
                    } catch (std::exception & ex) {
                        ALPS_HDF5_THROW_RUNTIME_ERROR("file: " + filename() + ", path: " + p + "\n" + ex.what());
                    }
                }
                void delete_group(std::string const & p) const {
                    try {
                        if (is_group(p))
                            // TODO: implement provenance
                            detail::check_error(H5Ldelete(file_id(), complete_path(p).c_str(), H5P_DEFAULT));
                        else
                            ALPS_HDF5_THROW_RUNTIME_ERROR("the path does not exists: " + p)
                    } catch (std::exception & ex) {
                        ALPS_HDF5_THROW_RUNTIME_ERROR("file: " + filename() + ", path: " + p + "\n" + ex.what());
                    }
                }
                void delete_attribute(std::string const & p) const {
                    try {
                        // TODO: implement
                    } catch (std::exception & ex) {
                        ALPS_HDF5_THROW_RUNTIME_ERROR("file: " + filename() + ", path: " + p + "\n" + ex.what());
                    }
                }
                std::vector<std::string> list_children(std::string const & p) const {
                    try {
                        std::vector<std::string> list;
                        if (!is_group(p))
                            ALPS_HDF5_THROW_RUNTIME_ERROR("The group '" + p + "' does not exists.")
                        detail::group_type group_id(H5Gopen2(file_id(), complete_path(p).c_str(), H5P_DEFAULT));
                        detail::check_error(H5Literate(group_id, H5_INDEX_NAME, H5_ITER_NATIVE, NULL, child_visitor, reinterpret_cast<void *>(&list)));
                        return list;
                    } catch (std::exception & ex) {
                        ALPS_HDF5_THROW_RUNTIME_ERROR("file: " + filename() + ", path: " + p + "\n" + ex.what());
                    }
                }
                std::vector<std::string> list_attr(std::string const & p) const {
                    try {
                        std::vector<std::string> list;
                        if (is_group(p)) {
                            detail::group_type id(H5Gopen2(file_id(), complete_path(p).c_str(), H5P_DEFAULT));
                            detail::check_error(H5Aiterate2(id, H5_INDEX_CRT_ORDER, H5_ITER_NATIVE, NULL, attr_visitor, reinterpret_cast<void *>(&list)));
                        } else if(is_data(p)) {
                            detail::data_type id(H5Dopen2(file_id(), complete_path(p).c_str(), H5P_DEFAULT));
                            detail::check_error(H5Aiterate2(id, H5_INDEX_CRT_ORDER, H5_ITER_NATIVE, NULL, attr_visitor, reinterpret_cast<void *>(&list)));
                        } else
                            ALPS_HDF5_THROW_RUNTIME_ERROR("The path '" + p + "' does not exists.")
                        return list;
                    } catch (std::exception & ex) {
                        ALPS_HDF5_THROW_RUNTIME_ERROR("file: " + filename() + ", path: " + p + "\n" + ex.what());
                    }
                }
            protected:
                archive(std::string const & filename, bool compress = false)
                    : _error_handler_id(H5Eset_auto2(H5E_DEFAULT, NULL, NULL))
                    , _hdf5_context(_pool.find(make_pair(filename, compress)) == _pool.end() || _pool[make_pair(filename, compress)].expired()
                        ? boost::shared_ptr<detail::hdf5_context>(new detail::hdf5_context(filename, F(filename), compress))
                        : _pool[make_pair(filename, compress)].lock()
                      )
                {
                    if (_pool.find(make_pair(filename, compress)) == _pool.end() || _pool[make_pair(filename, compress)].expired()) {
                        _pool[make_pair(filename, compress)] = _hdf5_context;
                        if (!is_group("/revisions")) {
                            hid_t group_id = H5Gcreate2(file_id(), "/revisions", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                            if (group_id > -1) {
                                detail::check_group(group_id);
                                set_attr("/revisions/@last", revision());
                                detail::internal_state_type::type v;
                                detail::type_type state_id = H5Tenum_create(H5T_NATIVE_SHORT);
                                detail::check_error(H5Tenum_insert(state_id, "CREATE", &(v = detail::internal_state_type::CREATE)));
                                detail::check_error(H5Tenum_insert(state_id, "PLACEHOLDER", &(v = detail::internal_state_type::PLACEHOLDER)));
                                detail::check_error(H5Tcommit2(file_id(), "state_type", state_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
                                detail::type_type log_id = H5Tcreate (H5T_COMPOUND, sizeof(detail::internal_log_type));
                                detail::type_type time_id(H5Tcopy(H5T_C_S1));
                                detail::check_error(H5Tset_size(time_id, H5T_VARIABLE));
                                detail::check_error(H5Tinsert(log_id, "time", HOFFSET(detail::internal_log_type, time), time_id));
                                detail::type_type name_id(H5Tcopy(H5T_C_S1));
                                detail::check_error(H5Tset_size(name_id, H5T_VARIABLE));
                                detail::check_error(H5Tinsert(log_id, "log", HOFFSET(detail::internal_log_type, name), name_id));
                                detail::check_error(H5Tcommit2(file_id(), "log_type", log_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
                            }
                        }
                        _hdf5_context->_complex_id = H5Tcreate (H5T_COMPOUND, sizeof(detail::internal_complex_type));
                        detail::check_error(H5Tinsert(complex_id(), "r", HOFFSET(detail::internal_complex_type, r), H5T_NATIVE_DOUBLE));
                        detail::check_error(H5Tinsert(complex_id(), "i", HOFFSET(detail::internal_complex_type, i), H5T_NATIVE_DOUBLE));
                        if (is_group("/revisions")) {
                            get_attr("/revisions/@last", _hdf5_context->_revision);
                            _hdf5_context->_log_id = detail::check_error(H5Topen2(file_id(), "log_type", H5P_DEFAULT));
                            _hdf5_context->_state_id = detail::check_error(H5Topen2(file_id(), "state_type", H5P_DEFAULT));
                        }
                    }
                }
                template<typename T> hid_t get_native_type(T) const {
                    ALPS_HDF5_THROW_RUNTIME_ERROR(std::string("no native type passed: ") + typeid(T).name())
                }
                hid_t get_native_type(char) const { return H5Tcopy(H5T_NATIVE_CHAR); }
                hid_t get_native_type(signed char) const { return H5Tcopy(H5T_NATIVE_SCHAR); }
                hid_t get_native_type(unsigned char) const { return H5Tcopy(H5T_NATIVE_UCHAR); }
                hid_t get_native_type(short) const { return H5Tcopy(H5T_NATIVE_SHORT); }
                hid_t get_native_type(unsigned short) const { return H5Tcopy(H5T_NATIVE_USHORT); }
                hid_t get_native_type(int) const { return H5Tcopy(H5T_NATIVE_INT); }
                hid_t get_native_type(unsigned) const { return H5Tcopy(H5T_NATIVE_UINT); }
                hid_t get_native_type(long) const { return H5Tcopy(H5T_NATIVE_LONG); }
                hid_t get_native_type(unsigned long) const { return H5Tcopy(H5T_NATIVE_ULONG); }
                hid_t get_native_type(long long) const { return H5Tcopy(H5T_NATIVE_LLONG); }
                hid_t get_native_type(unsigned long long) const { return H5Tcopy(H5T_NATIVE_ULLONG); }
                hid_t get_native_type(float) const { return H5Tcopy(H5T_NATIVE_FLOAT); }
                hid_t get_native_type(double) const { return H5Tcopy(H5T_NATIVE_DOUBLE); }
                hid_t get_native_type(long double) const { return H5Tcopy(H5T_NATIVE_LDOUBLE); }
                hid_t get_native_type(bool) const { return H5Tcopy(H5T_NATIVE_HBOOL); }
                #ifdef ALPS_HDF5_WRITE_PYTHON_COMPATIBLE_COMPLEX
                    template<typename T> hid_t get_native_type(std::complex<T>) const { return H5Tcopy(complex_id()); }
                #endif
                hid_t get_native_type(detail::internal_log_type) const { return H5Tcopy(log_id()); }
                hid_t get_native_type(std::string) const { 
                    hid_t type_id = H5Tcopy(H5T_C_S1);
                    detail::check_error(H5Tset_size(type_id, H5T_VARIABLE));
                    return type_id;
                }
                static herr_t child_visitor(hid_t, char const * n, const H5L_info_t *, void * d) {
                    reinterpret_cast<std::vector<std::string> *>(d)->push_back(n);
                    return 0;
                }
                static herr_t attr_visitor(hid_t, char const * n, const H5A_info_t *, void * d) {
                    reinterpret_cast<std::vector<std::string> *>(d)->push_back(n);
                    return 0;
                }
                hid_t create_path(std::string const & p, hid_t type_id, hid_t space_id, hsize_t d, hsize_t const * s = NULL, bool set_prop = true) const {
                    hid_t data_id = H5Dopen2(file_id(), p.c_str(), H5P_DEFAULT), tmp_id = 0;
                    if (data_id < 0) {
                        if (p.find_last_of('/') < std::string::npos && p.find_last_of('/') > 0)
                            set_group(p.substr(0, p.find_last_of('/')));
                        data_id = create_dataset(p, type_id, space_id, d, s, set_prop);
                    } else if (
                           (d > 0 && s[0] > 0 && is_null(p)) 
                        || (d > 0 && s[0] == 0 && !is_null(p)) 
                        || !detail::check_error(H5Tequal(detail::type_type(H5Dget_type(data_id)), detail::type_type(H5Tcopy(type_id))))
                        || (d > 0 && s[0] > 0 && H5Dset_extent(data_id, s) < 0)
                    ) {
                        std::vector<std::string> names = list_attr(p);
                        if (names.size()) {
                            tmp_id = H5Gcreate2(file_id(), "/revisions/waitingroom", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                            copy_attributes(tmp_id, data_id, names);
                        }
                        detail::check_data(data_id);
                        detail::check_error(H5Ldelete(file_id(), p.c_str(), H5P_DEFAULT));
                        data_id = create_dataset(p, type_id, space_id, d, s, set_prop);
                        if (names.size()) {
                            copy_attributes(data_id, tmp_id, names);
                            detail::check_group(tmp_id);
                            detail::check_error(H5Ldelete(file_id(), "/revisions/waitingroom", H5P_DEFAULT));
                        }
                    }
                    return data_id;
                }
                hid_t create_dataset(std::string const & p, hid_t type_id, hid_t space_id, hsize_t d, hsize_t const * s = NULL, bool set_prop = true) const {
                    if (set_prop) {
                        detail::property_type prop_id(H5Pcreate(H5P_DATASET_CREATE));
                        detail::check_error(H5Pset_fill_time(prop_id, H5D_FILL_TIME_NEVER));
                        if (d > 0) {
                            detail::check_error(H5Pset_chunk(prop_id, d, s));
                            std::size_t n = std::accumulate(s, s + d, std::size_t(0));
                            if (compress() && n > ALPS_HDF5_SZIP_BLOCK_SIZE)
                                detail::check_error(H5Pset_szip(prop_id, H5_SZIP_NN_OPTION_MASK, ALPS_HDF5_SZIP_BLOCK_SIZE));
                        }
                        return H5Dcreate2(file_id(), p.c_str(), type_id, detail::space_type(space_id), H5P_DEFAULT, prop_id, H5P_DEFAULT);
                    } else
                        return H5Dcreate2(file_id(), p.c_str(), type_id, detail::space_type(space_id), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                }
                void copy_attributes(hid_t dest_id, hid_t source_id, std::vector<std::string> const & names) const {
                    for (std::vector<std::string>::const_iterator it = names.begin(); it != names.end(); ++it) {
                        detail::attribute_type attr_id = H5Aopen(source_id, it->c_str(), H5P_DEFAULT);
                        detail::type_type type_id = H5Aget_type(attr_id);
                        if (H5Tget_class(type_id) == H5T_STRING) {
                            std::string v;
                            v.resize(H5Tget_size(type_id));
                            detail::check_error(H5Aread(attr_id, detail::type_type(H5Tcopy(type_id)), &v[0]));
                            detail::attribute_type new_id = H5Acreate2(dest_id, it->c_str(), type_id, detail::space_type(H5Screate(H5S_SCALAR)), H5P_DEFAULT, H5P_DEFAULT);
                            detail::check_error(H5Awrite(new_id, type_id, &v[0]));
                        } else if (detail::check_error(H5Tequal(detail::type_type(H5Tcopy(type_id)), detail::type_type(H5Tcopy(state_id())))) > 0) {
                            detail::internal_state_type::type v;
                            detail::check_error(H5Aread(attr_id, state_id(), &v));
                            detail::attribute_type new_id = H5Acreate2(dest_id, it->c_str(), state_id(), detail::space_type(H5Screate(H5S_SCALAR)), H5P_DEFAULT, H5P_DEFAULT);
                            detail::check_error(H5Awrite(new_id, state_id(), &v));
                        }
                        #define HDF5_COPY_ATTR(T)                                                                                                          \
                            else if (detail::check_error(                                                                                                  \
                                H5Tequal(detail::type_type(H5Tcopy(type_id)), detail::type_type(get_native_type(static_cast<T>(0))))                       \
                            ) > 0) {                                                                                                                       \
                                T v;                                                                                                                       \
                                detail::check_error(H5Aread(attr_id, detail::type_type(H5Tcopy(type_id)), &v));                                            \
                                detail::attribute_type new_id = H5Acreate2(                                                                                \
                                    dest_id, it->c_str(), type_id, detail::space_type(H5Screate(H5S_SCALAR)), H5P_DEFAULT, H5P_DEFAULT                     \
                                );                                                                                                                         \
                                detail::check_error(H5Awrite(new_id, type_id, &v));                                                                        \
                            }
                        HDF5_FOREACH_SCALAR(HDF5_COPY_ATTR)
                        #undef HDF5_COPY_ATTR
                        else ALPS_HDF5_THROW_RUNTIME_ERROR("error in copying attribute: " + *it)
                    }
                }
                hid_t save_comitted_data(std::string const & p, hid_t type_id, hid_t space_id, hsize_t d, hsize_t const * s = NULL, bool set_prop = true) const {
                    std::string rev_path = "/revisions/" + boost::lexical_cast<std::string>(revision()) + p;
                    if (revision() && !is_data(p))
                        set_data(rev_path, detail::internal_state_type::CREATE);
                    else if (revision()) {
                        hid_t data_id = H5Dopen2(file_id(), rev_path.c_str(), H5P_DEFAULT);
                        std::vector<std::string> revision_names;
                        if (data_id > 0 && detail::check_error(H5Tequal(detail::type_type(H5Dget_type(data_id)), detail::type_type(H5Tcopy(state_id())))) > 0) {
                            detail::internal_state_type::type v;
                            detail::check_error(H5Dread(data_id, state_id(), H5S_ALL, H5S_ALL, H5P_DEFAULT, &v));
                            if (v == detail::internal_state_type::PLACEHOLDER) {
                                if ((revision_names = list_attr(rev_path)).size()) {
                                    detail::group_type tmp_id = H5Gcreate2(file_id(), "/revisions/waitingroom", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                                    copy_attributes(tmp_id, data_id, revision_names);
                                }
                                detail::check_data(data_id);
                                detail::check_error(H5Ldelete(file_id(), rev_path.c_str(), H5P_DEFAULT));
                            } else
                                detail::check_data(data_id);
                        } else if (data_id >= 0)
                            detail::check_data(data_id);
                        if (!is_data(rev_path)) {
                            set_group(rev_path.substr(0, rev_path.find_last_of('/')));
                            detail::check_error(H5Lmove(file_id(), p.c_str(), H5L_SAME_LOC, (rev_path).c_str(), H5P_DEFAULT, H5P_DEFAULT));
                            hid_t new_id = create_path(p, type_id, space_id, d, s, set_prop);
                            std::vector<std::string> current_names = list_attr(rev_path);
                            detail::data_type data_id(H5Dopen2(file_id(), rev_path.c_str(), H5P_DEFAULT));
                            copy_attributes(new_id, data_id, current_names); 
                            for (std::vector<std::string>::const_iterator it = current_names.begin(); it != current_names.end(); ++it)
                                H5Adelete(data_id, it->c_str());
                            if (revision_names.size()) {
                                copy_attributes(data_id, detail::group_type(H5Gopen2(file_id(), "/revisions/waitingroom", H5P_DEFAULT)), revision_names);
                                detail::check_error(H5Ldelete(file_id(), "/revisions/waitingroom", H5P_DEFAULT));
                            }
                            return new_id;
                        }
                    }
                    return create_path(p, type_id, space_id, d, s, set_prop);
                }
                hid_t open_attribute(std::string const & p) const {
                    hid_t parent_id, attr_id;
                    if (p.find_last_of('@') == std::string::npos)
                        ALPS_HDF5_THROW_RUNTIME_ERROR("no valid attribute path " + p)
                    else if (is_group(p.substr(0, p.find_last_of('@') - 1)))
                        parent_id = detail::check_error(H5Gopen2(file_id(), p.substr(0, p.find_last_of('@') - 1).c_str(), H5P_DEFAULT));
                    else if (is_data(p.substr(0, p.find_last_of('@') - 1)))
                        parent_id = detail::check_error(H5Dopen2(file_id(), p.substr(0, p.find_last_of('@') - 1).c_str(), H5P_DEFAULT));
                    else
                        #ifdef ALPS_HDF5_READ_GREEDY
                            return false;
                        #else
                            ALPS_HDF5_THROW_RUNTIME_ERROR("unknown path: " + p.substr(0, p.find_last_of('@') - 1))
                        #endif
                    attr_id = H5Aopen(parent_id, p.substr(p.find_last_of('@') + 1).c_str(), H5P_DEFAULT);
                    if (is_group(p.substr(0, p.find_last_of('@') - 1)))
                        detail::check_group(parent_id);
                    else
                        detail::check_data(parent_id);
                    return attr_id;
                }
                template<typename T, typename U> void get_helper_read(T & v, hid_t data_id, hid_t type_id, bool is_attr) const {
                    std::vector<hsize_t> size(call_get_extent(v)), start(size.size(), 0), count(call_get_offset(v));
                    if (
                           std::equal(count.begin(), count.end(), size.begin())
                        && H5Tget_class(type_id) == H5T_STRING && !detail::check_error(H5Tis_variable_str(type_id))
                    ) {
                        std::string data(H5Tget_size(type_id) + 1, '\0');
                        detail::check_error(is_attr ? H5Aread(data_id, type_id, &data[0]) : H5Dread(data_id, type_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, &data[0]));
                        call_set_data(v, std::vector<char *>(1, &data[0]), start, count);
                    } else if (std::equal(count.begin(), count.end(), size.begin())) {
                        std::vector<U> data(std::accumulate(count.begin(), count.end(), std::size_t(1), std::multiplies<std::size_t>()));
                        detail::check_error(is_attr ? H5Aread(data_id, type_id, &data.front()) : H5Dread(data_id, type_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, &data.front()));
                        call_set_data(v, data, start, count);
                        if (!is_attr && boost::is_same<T, char *>::value)
                            detail::check_error(H5Dvlen_reclaim(type_id, detail::space_type(H5Dget_space(data_id)), H5P_DEFAULT, &data.front()));
                    } else if (is_attr) {
                        std::size_t last = count.size() - 1, pos;
                        for(;count[last] == size[last]; --last);
                        std::vector<U> data(std::accumulate(size.begin(), size.end(), hsize_t(1), std::multiplies<hsize_t>()));
                        std::vector<U> chunk(std::accumulate(count.begin(), count.end(), hsize_t(1), std::multiplies<hsize_t>()));
                        detail::check_error(H5Aread(data_id, type_id, &data.front()));
                        do {
                            std::size_t sum = 0;
                            for (std::vector<hsize_t>::const_iterator it = start.begin(); it != start.end(); ++it)
                                sum += *it * std::accumulate(size.begin() + (it - start.begin()) + 1, size.end(), hsize_t(1), std::multiplies<hsize_t>());
                            std::copy(
                                data.begin() + sum,
                                data.begin() + sum + chunk.size(),
                                chunk.begin()
                            );
                            call_set_data(v, chunk, start, count);
                            if (start[last] + 1 == size[last] && last) {
                                for (pos = last; ++start[pos] == size[pos] && pos; --pos);
                                for (++pos; pos <= last; ++pos)
                                    start[pos] = 0;
                            } else
                                ++start[last];
                            if (boost::is_same<T, char *>::value)
                                detail::check_error(H5Dvlen_reclaim(type_id, detail::space_type(H5Dget_space(data_id)), H5P_DEFAULT, &data.front()));
                        } while (start[0] < size[0]);
                    } else {
                        std::size_t last = count.size() - 1, pos;
                        for(;count[last] == size[last]; --last);
                        std::vector<U> data(std::accumulate(count.begin(), count.end(), std::size_t(1), std::multiplies<std::size_t>()));
                        do {
                            detail::space_type space_id(H5Dget_space(data_id));
                            detail::check_error(H5Sselect_hyperslab(space_id, H5S_SELECT_SET, &start.front(), NULL, &count.front(), NULL));
                            detail::space_type mem_id(H5Screate_simple(count.size(), &count.front(), NULL));
                            detail::check_error(H5Dread(data_id, type_id, mem_id, space_id, H5P_DEFAULT, &data.front()));
                            call_set_data(v, data, start, count);
                            if (boost::is_same<T, char *>::value)
                                detail::check_error(H5Dvlen_reclaim(type_id, mem_id, H5P_DEFAULT, &data.front()));
                            if (start[last] + 1 == size[last] && last) {
                                for (pos = last; ++start[pos] == size[pos] && pos; --pos);
                                for (++pos; pos <= last; ++pos)
                                    start[pos] = 0;
                            } else
                                ++start[last];
                        } while (start[0] < size[0]);
                    }
                }
                template<typename T, typename I> void get_helper(std::string const & p, T & v, bool is_attr) const {
                    if (is_scalar(p) != is_native<T>::value)
                        ALPS_HDF5_THROW_RUNTIME_ERROR("scalar - vector conflict in path: " + p)
                    else if (is_native<T>::value && is_null(p))
                        ALPS_HDF5_THROW_RUNTIME_ERROR("scalars cannot be null in path: " + p)
                    else if (is_null(p))
                        call_set_extent(v, std::vector<std::size_t>(1, 0));
                    else {
                        std::vector<hsize_t> size(dimensions(p), 0);
                        I data_id(is_attr ? open_attribute(p) : H5Dopen2(file_id(), p.c_str(), H5P_DEFAULT));
                        detail::type_type type_id(is_attr ? H5Aget_type(data_id) : H5Dget_type(data_id));
                        detail::type_type native_id(H5Tget_native_type(type_id, H5T_DIR_ASCEND));
                        if (size.size()) {
                            detail::space_type space_id(is_attr ? H5Aget_space(data_id) : H5Dget_space(data_id));
                            detail::check_error(H5Sget_simple_extent_dims(space_id, &size.front(), NULL));
                        }
                        call_set_extent(v, std::vector<std::size_t>(size.begin(), size.end()));
                        if (H5Tget_class(native_id) == H5T_STRING)
                            get_helper_read<T, char *>(v, data_id, type_id, is_attr);
                        else if (detail::check_error(H5Tequal(detail::type_type(H5Tcopy(complex_id())), detail::type_type(H5Tcopy(type_id)))))
                            get_helper_read<T, std::complex<double> >(v, data_id, type_id, is_attr);
                        #define HDF5_GET_STRING(U)                                                                                                          \
                            else if (detail::check_error(                                                                                                   \
                                H5Tequal(detail::type_type(H5Tcopy(native_id)), detail::type_type(get_native_type(static_cast<U>(0))))                      \
                            ) > 0)                                                                                                                          \
                                get_helper_read<T, U>(v, data_id, type_id, is_attr);
                        HDF5_FOREACH_SCALAR(HDF5_GET_STRING)
                        #undef HDF5_GET_STRING
                        else ALPS_HDF5_THROW_RUNTIME_ERROR("invalid type")
                    }
                }
                template<typename T> void get_data(std::string const & p, T & v) const {
                    get_helper<T, detail::data_type>(p, v, false);
                }
                template<typename T> void get_attr(std::string const & p, T & v) const {
                    get_helper<T, detail::attribute_type>(p, v, true);
                }
                template<typename T> void set_data(std::string const & p, T const & v) const {
                    if (is_group(p))
                        delete_group(p);
                    detail::type_type type_id(get_native_type(typename native_type<T>::type()));
                    std::vector<hsize_t> size(call_get_extent(v)), start(size.size(), 0), count(call_get_offset(v));
                    std::vector<typename serializable_type<T>::type> data;
                    if (is_native<T>::value) {
                        detail::data_type data_id(save_comitted_data(p, type_id, H5Screate(H5S_SCALAR), 0, NULL, !boost::is_same<typename native_type<T>::type, std::string>::value));
                        detail::check_error(H5Dwrite(data_id, type_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, call_get_data(data, v, start)));
                    } else if (std::accumulate(size.begin(), size.end(), hsize_t(0)) == 0)
                        detail::check_data(save_comitted_data(p, type_id, H5Screate(H5S_NULL), 0, NULL, !boost::is_same<typename native_type<T>::type, std::string>::value));
                    else {
                        detail::data_type data_id(save_comitted_data(p, type_id, H5Screate_simple(size.size(), &size.front(), NULL), size.size(), &size.front(), !boost::is_same<typename native_type<T>::type, std::string>::value));
                        if (std::equal(count.begin(), count.end(), size.begin()))
                            detail::check_error(H5Dwrite(data_id, type_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, call_get_data(data, v, start)));
                        else {
                            std::size_t last = count.size() - 1, pos;
                            for(;count[last] == size[last]; --last);
                            do {
                                detail::space_type space_id(H5Dget_space(data_id));
                                detail::check_error(H5Sselect_hyperslab(space_id, H5S_SELECT_SET, &start.front(), NULL, &count.front(), NULL));
                                detail::space_type mem_id(H5Screate_simple(count.size(), &count.front(), NULL));
                                detail::check_error(H5Dwrite(data_id, type_id, mem_id, space_id, H5P_DEFAULT, call_get_data(data, v, start)));
                                if (start[last] + 1 == size[last] && last) {
                                    for (pos = last; ++start[pos] == size[pos] && pos; --pos);
                                    for (++pos; pos <= last; ++pos)
                                        start[pos] = 0;
                                } else
                                    ++start[last];
                            } while (start[0] < size[0]);
                        }
                    }
                }
                template<typename T> void set_attr(std::string const & p, T const & v) const {
                    hid_t parent_id;
                    std::string rev_path = "/revisions/" + boost::lexical_cast<std::string>(revision()) + p;
                    if (is_group(p.substr(0, p.find_last_of('@') - 1))) {
                        parent_id = detail::check_error(H5Gopen2(file_id(), p.substr(0, p.find_last_of('@') - 1).c_str(), H5P_DEFAULT));
                        if (revision() && p.substr(0, p.find_last_of('@') - 1).substr(0, std::strlen("/revisions")) != "/revisions" && !is_group(rev_path))
                            set_group(rev_path);
                    } else if (is_data(p.substr(0, p.find_last_of('@') - 1))) {
                        parent_id = detail::check_error(H5Dopen2(file_id(), p.substr(0, p.find_last_of('@') - 1).c_str(), H5P_DEFAULT));
                        if (revision() && p.substr(0, p.find_last_of('@') - 1).substr(0, std::strlen("/revisions")) != "/revisions" && !is_data(rev_path))
                            set_data(rev_path, detail::internal_state_type::PLACEHOLDER);
                    } else
                        ALPS_HDF5_THROW_RUNTIME_ERROR("unknown path: " + p.substr(0, p.find_last_of('@') - 1))
                    if (revision() 
                        && p.substr(0, p.find_last_of('@') - 1).substr(0, std::strlen("/revisions")) 
                           != "/revisions" && !detail::check_error(H5Aexists(parent_id, p.substr(p.find_last_of('@') + 1).c_str()))
                    )
                        set_attr(rev_path + "/@" + p.substr(p.find_last_of('@') + 1), detail::internal_state_type::CREATE);
                    else if (revision() && p.substr(0, p.find_last_of('@') - 1).substr(0, std::strlen("/revisions")) != "/revisions") {
                        hid_t data_id = (is_group(rev_path) ? H5Gopen2(file_id(), rev_path.c_str(), H5P_DEFAULT) : H5Dopen2(file_id(), rev_path.c_str(), H5P_DEFAULT));
                        if (detail::check_error(H5Aexists(data_id, p.substr(p.find_last_of('@') + 1).c_str())) 
                            && detail::check_error(
                                  H5Tequal(detail::type_type(H5Aget_type(detail::attribute_type(H5Aopen(data_id, p.substr(p.find_last_of('@') + 1).c_str(), H5P_DEFAULT))))
                                , detail::type_type(H5Tcopy(state_id())))
                               ) > 0
                        )
                            H5Adelete(data_id, p.substr(p.find_last_of('@') + 1).c_str());
                        if (!detail::check_error(H5Aexists(data_id, p.substr(p.find_last_of('@') + 1).c_str())))
                            copy_attributes(data_id, parent_id, std::vector<std::string>(1, p.substr(p.find_last_of('@') + 1)));
                        if (is_group(p.substr(0, p.find_last_of('@') - 1)))
                            detail::check_group(data_id);
                        else
                            detail::check_data(data_id);
                    }
                    detail::type_type type_id(get_native_type(typename native_type<T>::type()));
                    std::vector<hsize_t> size(call_get_extent(v)), start(size.size(), 0), count(call_get_offset(v));
                    std::vector<typename serializable_type<T>::type> data;
                    hid_t id = H5Aopen(parent_id, p.substr(p.find_last_of('@') + 1).c_str(), H5P_DEFAULT);
                    if (id > 0) {
                        detail::space_type space_id(H5Aget_space(id));
                        H5S_class_t type = H5Sget_simple_extent_type(space_id);
                        if (type == H5S_NO_CLASS)
                            ALPS_HDF5_THROW_RUNTIME_ERROR("error reading class " + p)
                        if ((size.size() > 0 && size[0] > 0 && type != H5S_SCALAR)
                            || (size.size() > 0 && size[0] == 0 && type != H5S_NULL)
                            || !detail::check_error(H5Tequal(detail::type_type(H5Aget_type(id)), detail::type_type(H5Tcopy(type_id))))
                        ) {
                            detail::check_attribute(id);
                            H5Adelete(parent_id, p.substr(p.find_last_of('@') + 1).c_str());
                            id = -1;
                        }
                    }
                    if (is_native<T>::value) {
                        if (id < 0)
                            id = H5Acreate2(parent_id, p.substr(p.find_last_of('@') + 1).c_str(), type_id, detail::space_type(H5Screate(H5S_SCALAR)), H5P_DEFAULT, H5P_DEFAULT);
                        detail::check_error(H5Awrite(id, type_id, call_get_data(data, v, start)));
                    } else if (std::accumulate(size.begin(), size.end(), hsize_t(0)) == 0) {
                        if (id < 0)
                            id = H5Acreate2(parent_id, p.substr(p.find_last_of('@') + 1).c_str(), type_id, detail::space_type(H5Screate(H5S_NULL)), H5P_DEFAULT, H5P_DEFAULT);
                    } else {
                        if (id < 0)
                            id = H5Acreate2(parent_id, p.substr(p.find_last_of('@') + 1).c_str(), type_id, detail::space_type(H5Screate_simple(size.size(), &size.front(), NULL)), H5P_DEFAULT, H5P_DEFAULT);
                        if (std::equal(count.begin(), count.end(), size.begin()))
                            detail::check_error(H5Awrite(id, type_id, call_get_data(data, v, start)));
                        else {
                            std::vector<typename native_type<T>::type> continous(std::accumulate(size.begin(), size.end(), hsize_t(1), std::multiplies<hsize_t>()));
                            std::size_t last = count.size() - 1, pos;
                            for(;count[last] == size[last]; --last);
                            do {
                                std::size_t sum = 0;
                                for (std::vector<hsize_t>::const_iterator it = start.begin(); it != start.end(); ++it)
                                    sum += *it * std::accumulate(size.begin() + (it - start.begin()) + 1, size.end(), hsize_t(1), std::multiplies<hsize_t>());
                                std::memcpy(
                                    &continous.front() + sum,
                                    call_get_data(data, v, start),
                                    std::accumulate(count.begin(), count.end(), hsize_t(1), std::multiplies<hsize_t>()) * sizeof(typename native_type<T>::type)
                                );
                                if (start[last] + 1 == size[last] && last) {
                                    for (pos = last; ++start[pos] == size[pos] && pos; --pos);
                                    for (++pos; pos <= last; ++pos)
                                        start[pos] = 0;
                                } else
                                    ++start[last];
                            } while (start[0] < size[0]);
                            detail::check_error(H5Awrite(id, type_id, &continous.front()));
                        }
                    }
                    detail::attribute_type attr_id(id);
                    if (is_group(p.substr(0, p.find_last_of('@') - 1)))
                        detail::check_group(parent_id);
                    else
                        detail::check_data(parent_id);
                }
                void set_group(std::string const & p) const {
                    if (!is_group(p)) {
                        std::size_t pos;
                        hid_t group_id = -1;
                        for (pos = p.find_last_of('/'); group_id < 0 && pos > 0 && pos < std::string::npos; pos = p.find_last_of('/', pos - 1))
                            group_id = H5Gopen2(file_id(), p.substr(0, pos).c_str(), H5P_DEFAULT);
                        if (group_id < 0) {
                            if ((pos = p.find_first_of('/', 1)) != std::string::npos)
                                detail::check_group(H5Gcreate2(file_id(), p.substr(0, pos).c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
                        } else {
                            pos = p.find_first_of('/', pos + 1);
                            detail::check_group(group_id);
                        }
                        while (pos != std::string::npos && (pos = p.find_first_of('/', pos + 1)) != std::string::npos && pos > 0)
                            detail::check_group(H5Gcreate2(file_id(), p.substr(0, pos).c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
                        detail::check_group(H5Gcreate2(file_id(), p.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
                    }
                }
            private:
                inline int compress() const {
                    return _hdf5_context->_compress;
                }
                inline int revision() const {
                    return _hdf5_context->_revision;
                }
                inline hid_t state_id() const {
                    return _hdf5_context->_state_id;
                }
                inline hid_t log_id() const {
                    return _hdf5_context->_log_id;
                }
                inline hid_t complex_id() const {
                    return _hdf5_context->_complex_id;
                }
                inline hid_t file_id() const {
                    return _hdf5_context->_file_id;
                }
                std::string _path_context;
                detail::error_type _error_handler_id;
                boost::shared_ptr<detail::hdf5_context> _hdf5_context;
                static std::map<std::pair<std::string, bool>, boost::weak_ptr<detail::hdf5_context> > _pool;
        };
        template<hid_t(* F )(std::string const &)> std::map<std::pair<std::string, bool>, boost::weak_ptr<detail::hdf5_context> > archive<F>::_pool;
        #undef HDF5_FOREACH_SCALAR

        namespace detail {
            struct creator {
                static hid_t open_reading(std::string const & file) {
                    if (!boost::filesystem::exists(file))
                        ALPS_HDF5_THROW_RUNTIME_ERROR("file does not exists: " + file)
                    if (detail::check_error(H5Fis_hdf5(file.c_str())) == 0)
                        ALPS_HDF5_THROW_RUNTIME_ERROR("no valid hdf5 file: " + file)
                    return H5Fopen(file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
                }
                static hid_t open_writing(std::string const & file) {
                    hid_t file_id = H5Fopen(file.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
                    return file_id < 0 ? H5Fcreate(file.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT) : file_id;
                }
            };
        }

        class iarchive : public archive<&detail::creator::open_reading> {
            public:
                iarchive(std::string const & file) : archive<&detail::creator::open_reading>(file) {}
                iarchive(iarchive const & rhs) : archive<&detail::creator::open_reading>(rhs) {}
                template<typename T> void serialize(std::string const & p, T & v) const {
                    if (p.find_last_of('@') != std::string::npos) {
                        #ifdef ALPS_HDF5_READ_GREEDY
                            if (is_attribute(p))
                        #endif
                                get_attr(complete_path(p), v);
                    } else
                        #ifdef ALPS_HDF5_READ_GREEDY
                            if (is_data(p))
                        #endif
                                get_data(complete_path(p), v);
                }
        };

        class oarchive : public archive<&detail::creator::open_writing> {
            public:
                oarchive(std::string const & file, bool compress = false) : archive<&detail::creator::open_writing>(file, compress) {}
                oarchive(oarchive const & rhs) : archive<&detail::creator::open_writing>(rhs) {}
                template<typename T> void serialize(std::string const & p, T const & v) const {
                    if (p.find_last_of('@') != std::string::npos)
                        set_attr(complete_path(p), v);
                    else
                        set_data(complete_path(p), v);
                }
                void serialize(std::string const & p) {
                    if (p.find_last_of('@') != std::string::npos)
                        ALPS_HDF5_THROW_RUNTIME_ERROR("attributes needs to be a scalar type or a string" + p)
                    else
                        set_group(complete_path(p));
                }
        };

        namespace detail {
            template<typename A, typename T> A & serialize_impl(A & ar, std::string const & p, T & v, boost::mpl::true_) {
                ar.serialize(p, v);
                return ar;
            }
            template<typename A, typename T> A & serialize_impl(A & ar, std::string const & p, T & v, boost::mpl::false_) {
                std::string c = ar.get_context();
                ar.set_context(ar.complete_path(p));
                v.serialize(ar);
                ar.set_context(c);
                return ar;
            }
        }

        template<typename T> iarchive & call_serialize(iarchive & ar, std::string const & p, T & v);
        template<typename T> oarchive & call_serialize(oarchive & ar, std::string const & p, T const & v);

        template<typename A, typename T> A & serialize(A & ar, std::string const & p, T & v) {
            return detail::serialize_impl(ar, p, v, is_native<T>());
        }

        template<typename T> iarchive & serialize(iarchive & ar, std::string const & p, std::complex<T> & v) {
            ar.serialize(p, v);
            return ar;
        }
        template<typename T> oarchive & serialize(oarchive & ar, std::string const & p, std::complex<T> const & v) {
            ar.serialize(p, v);
            return ar;
        }

        #define HDF5_DEFINE_VECTOR_TYPE(C)                                                                                                                  \
            template<typename T> iarchive & serialize(iarchive & ar, std::string const & p, C <T> & v) {                                                    \
                if (ar.is_group(p)) {                                                                                                                       \
                    std::vector<std::string> children = ar.list_children(p);                                                                                \
                    v.resize(children.size());                                                                                                              \
                    for (std::vector<std::string>::const_iterator it = children.begin(); it != children.end(); ++it)                                        \
                        call_serialize(ar, p + "/" + *it, v[boost::lexical_cast<std::size_t>(*it)]);                                                        \
                } else                                                                                                                                      \
                    ar.serialize(p, v);                                                                                                                     \
                return ar;                                                                                                                                  \
            }                                                                                                                                               \
            template<typename T> oarchive & serialize(oarchive & ar, std::string const & p, C <T> const & v) {                                              \
                if (ar.is_group(p))                                                                                                                         \
                    ar.delete_group(p);                                                                                                                     \
                if (!v.size())                                                                                                                              \
                    ar.serialize(p, C <int>(0));                                                                                                            \
                else if (call_is_vectorizable(v))                                                                                                           \
                    ar.serialize(p, v);                                                                                                                     \
                else {                                                                                                                                      \
                    if (p.find_last_of('@') != std::string::npos)                                                                                           \
                        ALPS_HDF5_THROW_RUNTIME_ERROR("attributes needs to be vectorizable: " + ar.complete_path(p))                                        \
                    else if (ar.is_data(p))                                                                                                                 \
                        ar.delete_data(p);                                                                                                                  \
                    for (std::size_t i = 0; i < v.size(); ++i)                                                                                              \
                        call_serialize(ar, p + "/" + boost::lexical_cast<std::string>(i), v[i]);                                                            \
                }                                                                                                                                           \
                return ar;                                                                                                                                  \
            }
        HDF5_DEFINE_VECTOR_TYPE(std::vector)
        HDF5_DEFINE_VECTOR_TYPE(std::valarray)
        HDF5_DEFINE_VECTOR_TYPE(boost::numeric::ublas::vector)
        #undef HDF5_DEFINE_VECTOR_TYPE

        template<typename T> iarchive & serialize(iarchive & ar, std::string const & p, std::deque<T> & v) {
            std::vector<T> d;
            call_serialize(ar, p, d);
            v.resize(d.size());
            std::copy(d.begin(), d.end(), v.begin());
            return ar;
        }
        template<typename T> oarchive & serialize(oarchive & ar, std::string const & p, std::deque<T> const & v) {
            std::vector<T> d(v.begin(), v.end());
            call_serialize(ar, p, d);
            return ar;
        }

        template<typename T> iarchive & serialize(iarchive & ar, std::string const & p, std::pair<T *, std::vector<std::size_t> > & v) {
            if (ar.is_group(p)) {
                std::vector<std::size_t> start(v.second.size(), 0);
                do {
                    std::size_t last = start.size() - 1, pos = 0;
                    std::string path = "";
                    for (std::vector<std::size_t>::const_iterator it = start.begin(); it != start.end(); ++it) {
                        path += "/" + boost::lexical_cast<std::string>(*it);
                        pos += *it * std::accumulate(v.second.begin() + (it - start.begin()) + 1, v.second.end(), std::size_t(1), std::multiplies<std::size_t>());
                    }
                    call_serialize(ar, p + path, v.first[pos]);
                    if (start[last] + 1 == v.second[last] && last) {
                        for (pos = last; ++start[pos] == v.second[pos] && pos; --pos);
                        for (++pos; pos <= last; ++pos)
                            start[pos] = 0;
                    } else
                        ++start[last];
                } while (start[0] < v.second[0]);
            } else
                detail::serialize_impl(ar, p, v, boost::mpl::true_());
            return ar;
        }
        template<typename T> oarchive & serialize(oarchive & ar, std::string const & p, std::pair<T *, std::vector<std::size_t> > const & v) {
            if (ar.is_group(p))
                ar.delete_group(p);
            if (!v.second.size())
               ar.serialize(p, make_pair(static_cast<int *>(NULL), v.second));
            else if (call_is_vectorizable(v))
                detail::serialize_impl(ar, p, v, boost::mpl::true_());
            else {
                if (p.find_last_of('@') != std::string::npos)
                    ALPS_HDF5_THROW_RUNTIME_ERROR("attributes needs to be vectorizable: " + ar.complete_path(p))
                if (ar.is_data(p))
                    ar.delete_data(p);
                std::vector<std::size_t> start(v.second.size(), 0);
                do {
                    std::size_t last = start.size() - 1, pos = 0;
                    std::string path = "";
                    for (std::vector<std::size_t>::const_iterator it = start.begin(); it != start.end(); ++it) {
                        path += "/" + boost::lexical_cast<std::string>(*it);
                        pos += *it * std::accumulate(v.second.begin() + (it - start.begin()) + 1, v.second.end(), std::size_t(1), std::multiplies<std::size_t>());
                    }
                    call_serialize(ar, p + path, v.first[pos]);
                    if (start[last] + 1 == v.second[last] && last) {
                        for (pos = last; ++start[pos] == v.second[pos] && pos; --pos);
                        for (++pos; pos <= last; ++pos)
                            start[pos] = 0;
                    } else
                        ++start[last];
                } while (start[0] < v.second[0]);
            }
            return ar;
        }

        template<typename T, std::size_t N, typename A> iarchive & serialize(iarchive & ar, std::string const & p, boost::multi_array<T, N, A> & v) {
            std::pair<T *, std::vector<std::size_t> > d(v.data(), std::vector<std::size_t>(v.shape(), v.shape() + boost::multi_array<T, N, A>::dimensionality));
            call_serialize(ar, p, d);
            return ar;
        }
        template<typename T, std::size_t N, typename A> oarchive & serialize(oarchive & ar, std::string const & p, boost::multi_array<T, N, A> const & v) {
            std::pair<T const *, std::vector<std::size_t> > d(v.data(), std::vector<std::size_t>(v.shape(), v.shape() + boost::multi_array<T, N, A>::dimensionality));
            call_serialize(ar, p, d);
            return ar;
        }

        template<typename T> iarchive & serialize(iarchive & ar, std::string const & p, boost::numeric::ublas::matrix<T ,boost::numeric::ublas::column_major> & v) {
            std::pair<T *, std::vector<std::size_t> > d(&v(0,0), std::vector<std::size_t>(2, v.size1()));
            d.second[1] = v.size2();
            detail::serialize_impl(ar, p, d, boost::mpl::true_());
            return ar;
        }
        template<typename T> oarchive & serialize(oarchive & ar, std::string const & p, boost::numeric::ublas::matrix<T ,boost::numeric::ublas::column_major> const & v) {
            std::pair<T const *, std::vector<std::size_t> > d(&v(0,0), std::vector<std::size_t>(2, v.size1()));
            d.second[1] = v.size2();
            detail::serialize_impl(ar, p, d, boost::mpl::true_());
            return ar;
        }

        template<typename T> iarchive & call_serialize(iarchive & ar, std::string const & p, T & v) { 
            return serialize(ar, p, v); 
        }
        template<typename T> oarchive & call_serialize(oarchive & ar, std::string const & p, T const & v) { 
            return serialize(ar, p, v); 
        }
        template <typename T> class pvp {
            public:
                pvp(std::string const & p, T v): _p(p), _v(v) {}
                pvp(pvp<T> const & c): _p(c._p), _v(c._v) {}
                template<typename A> A & serialize(A & ar) const {
                    try {
                        return call_serialize(ar, _p, const_cast<typename boost::add_reference<T>::type>(_v));
                    } catch (std::exception & ex) {
                        throw std::runtime_error("HDF5 Error " + std::string(boost::is_same<A, iarchive>::value ? "reading" : "writing") + " path '" + _p + "' on type '" + typeid(_v).name() + "':\n" + ex.what());
                    }
                }
            private:
                std::string _p;
                T _v;
        };

        template <typename T> iarchive & operator>> (iarchive & ar, pvp<T> const & v) { return v.serialize(ar); }
        template <typename T> oarchive & operator<< (oarchive & ar, pvp<T> const & v) { return v.serialize(ar); }
    }

    template <typename T> typename boost::disable_if<typename boost::mpl::and_<
          typename boost::is_same<typename boost::remove_cv<typename boost::remove_all_extents<T>::type>::type, char>::type
        , typename boost::is_array<T>::type
    >::type, hdf5::pvp<T &> >::type make_pvp(std::string const & p, T & v) {
        return hdf5::pvp<T &>(p, v);
    }
    template <typename T> typename boost::disable_if<typename boost::mpl::and_<
          typename boost::is_same<typename boost::remove_cv<typename boost::remove_all_extents<T>::type>::type, char>::type
        , typename boost::is_array<T>::type
    >::type, hdf5::pvp<T const &> >::type make_pvp(std::string const & p, T const & v) {
        return hdf5::pvp<T const &>(p, v);
    }
    template <typename T> typename boost::enable_if<typename boost::mpl::and_<
          typename boost::is_same<typename boost::remove_cv<typename boost::remove_all_extents<T>::type>::type, char>::type
        , typename boost::is_array<T>::type
    >::type, hdf5::pvp<std::string const> >::type make_pvp(std::string const & p, T const & v) {
        return hdf5::pvp<std::string const>(p, v);
    }
    
    template <typename T> hdf5::pvp<std::pair<T *, std::vector<std::size_t> > > make_pvp(std::string const & p, T * v, std::size_t s) {
        using namespace boost;
        return hdf5::pvp<std::pair<T *, std::vector<std::size_t> > >(p, std::make_pair(ref(v), std::vector<std::size_t>(1, s)));
    }
    template <typename T> hdf5::pvp<std::pair<T *, std::vector<std::size_t> > > make_pvp(std::string const & p, T * v, std::vector<std::size_t> const & s) {
        using namespace boost;
        return hdf5::pvp<std::pair<T *, std::vector<std::size_t> > >(p, std::make_pair(ref(v), s));
    }

    #define HDF5_MAKE_PVP(ptr_type, arg_type)                                                                                                               \
        template <typename T> hdf5::pvp<std::pair<ptr_type, std::vector<std::size_t> > > make_pvp(std::string const & p, arg_type v, std::size_t s) {       \
            return hdf5::pvp<std::pair<ptr_type, std::vector<std::size_t> > >(p, std::make_pair(&*v, std::vector<std::size_t>(1, s)));                      \
        }                                                                                                                                                   \
        template <typename T> hdf5::pvp<std::pair<ptr_type, std::vector<std::size_t> > > make_pvp(                                                          \
            std::string const & p, arg_type v, std::vector<std::size_t> const & s                                                                           \
        ) {                                                                                                                                                 \
            return hdf5::pvp<std::pair<ptr_type, std::vector<std::size_t> > >(p, std::make_pair(&*v, s));                                                   \
        }
    HDF5_MAKE_PVP(T *, boost::shared_ptr<T> &)
    HDF5_MAKE_PVP(T const *, boost::shared_ptr<T> const &)
    HDF5_MAKE_PVP(T *, std::auto_ptr<T> &)
    HDF5_MAKE_PVP(T const *, std::auto_ptr<T> const &)
    HDF5_MAKE_PVP(T *, boost::weak_ptr<T> &)
    HDF5_MAKE_PVP(T const *, boost::weak_ptr<T> const &)
    HDF5_MAKE_PVP(T *, boost::scoped_ptr<T> &)
    HDF5_MAKE_PVP(T const *, boost::scoped_ptr<T> const &)
    #undef HDF5_MAKE_PVP

    #define HDF5_MAKE_ARRAY_PVP(ptr_type, arg_type)                                                                                                         \
        template <typename T> hdf5::pvp<std::pair<ptr_type, std::vector<std::size_t> > > make_pvp(std::string const & p, arg_type v, std::size_t s) {       \
            return hdf5::pvp<std::pair<ptr_type, std::vector<std::size_t> > >(p, std::make_pair(v.get(), std::vector<std::size_t>(1, s)));                  \
        }                                                                                                                                                   \
        template <typename T> hdf5::pvp<std::pair<ptr_type, std::vector<std::size_t> > > make_pvp(                                                          \
            std::string const & p, arg_type v, std::vector<std::size_t> const & s                                                                           \
        ) {                                                                                                                                                 \
            return hdf5::pvp<std::pair<ptr_type, std::vector<std::size_t> > >(p, std::make_pair(v.get(), s));                                               \
        }
    HDF5_MAKE_ARRAY_PVP(T *, boost::shared_array<T> &)
    HDF5_MAKE_ARRAY_PVP(T const *, boost::shared_array<T> const &)
    #undef HDF5_MAKE_ARRAY_PVP

    #undef ALPS_HDF5_STRINGIFY
    #undef ALPS_HDF5_STRINGIFY_HELPER
    #undef ALPS_HDF5_THROW_ERROR
    #undef ALPS_HDF5_THROW_RUNTIME_ERROR
    #undef ALPS_HDF5_THROW_RANGE_ERROR
}
#endif
