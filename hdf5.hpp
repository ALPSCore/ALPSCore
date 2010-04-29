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
#include <stdexcept>
#include <valarray>
#include <iostream>

#include <boost/any.hpp>
#include <boost/array.hpp>
#include <boost/config.hpp>
#include <boost/mpl/or.hpp>
#include <boost/mpl/if.hpp>
#include <boost/utility.hpp>
#include <boost/mpl/and.hpp>
#include <boost/optional.hpp>
#include <boost/weak_ptr.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/multi_array.hpp>
#include <boost/type_traits.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/static_assert.hpp>
#include <boost/intrusive_ptr.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/date_time/gregorian/gregorian.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include <hdf5.h>

#ifdef ALPS_DOXYGEN
  
namespace alps {
    namespace hdf5 {
        namespace detail {
            UNSPECIFIED_TYPE unspecified_type;
        }
        struct write {};
        struct read {};
        template <typename Tag> class archive: boost::noncopyable {
            public:
                /// @file path to the hdf5 file to build the archive from. In case of iarchive, the file is opend in read only mode
                archive(std::string const & file);
                ~archive();
                /// @return return the filename of the file, the arive ist based on
                std::string const & filename() const;
                std::string encode_segment(std::string const & s);
                std::string decode_segment(std::string const & s);
                /// create a checkpoint of the data.
                void commit(std::string const & name = "");
                /// @return list of all checkpoints with name and time 
                std::vector<std::pair<std::string, std::string> > list_revisions() const;
                /// export a checkpoint to a separat file
                void export_revision(std::size_t revision, std::string const & file) const;
                /// get the current context of the archive
                std::string get_context() const;
                /// set the context of the archive
                void set_context(std::string const & context);
                /// compute the absolte path given a path relative to the context
                std::string compute_path(std::string const & path) const;
                /// checks if a group is located at the given path
                bool is_group(std::string const & path) const;
                /// checks if a dataset is located at the given path
                bool is_data(std::string const & path) const;
                /// checks if a dataset containing a scalar is located at the given path
                bool is_scalar(std::string const & path) const;
                /// checks if a dataset containing a null pinter is located at the given path
                bool is_null(std::string const & path) const;
                /// checks if a attribute located at the given path. An attribute is addressed with a path of the form /path/to/@attribute
                bool is_attribute(std::string const & path) const;
                /// deletes a dataset
                void delete_data(std::string const & path) const;
                /// @return extents of the dataset located at the given path. extend(...).size() == dimensions(...) always holds
                std::vector<std::size_t> extent(std::string const & path) const;
                /// number of dimensions of the dataset located at the given path
                std::size_t dimensions(std::string const & path) const;
                /// list of all child segments of the given path
                std::vector<std::string> list_children(std::string const & path) const;
                /// list of all attributes of the fiven path
                std::vector<std::string> list_attr(std::string const & path) const;
                /// write data to archive
                template<typename T> void serialize(std::string const & p, T const & v);
                /// read data from archive
                template<typename T> void serialize(std::string const & p, T & v);
                /// create group at given path
                void serialize(std::string const & p);
        };
        /// input archive
        typedef archive<read> iarchive;
        /// output archive
        typedef archive<write> oarchive;
        /// global hook to deserialize an arbitrary object less intrusive
        template <typename T> iarchive & serialize(iarchive & archive, std::string const & path, T & value);
        /// global hook to serialize an arbitrary object less intrusive
        template <typename T> oarchive & serialize(oarchive & archive, std::string const & path, T const & value);
        /// operator to serialize data to hdf5
        template <typename T> oarchive & operator<< (oarchive & archive, unspecified_type value);
        /// operator to deserialize data from hdf5
        template <typename T> iarchive & operator>> (iarchive & archive, unspecified_type value);
        /// create a path-value-pair
        template <typename T> unspecified_type make_pvp(std::string const & path, T value);
        /// create a path-value-pair
        template <typename T> unspecified_type make_pvp(std::string const & path, T * value, std::size_t size);
        /// create a path-value-pair
        template <typename T> unspecified_type make_pvp(std::string const & path, T * value, std::vector<std::size_t> size);
    }
}

#else

namespace alps {
    namespace hdf5 {
        namespace detail {
            struct write {};
            struct read {};
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
            template<typename T> struct matrix_type : public boost::is_scalar<T>::type {
                typedef T native_type;
                typedef T const * pointer_type;
                typedef T const * buffer_type;
                typedef boost::mpl::true_ scalar;
                // TODO: start, count und size sollten auch fuer die skalardimension einen Eintrag haben also vector<int>(5) hat size [5, 1]
                static std::vector<hsize_t> count(T const & v) { return size(v); }
                static std::vector<hsize_t> size(T const & v) { return std::vector<hsize_t>(1, 1); }
                static pointer_type get(buffer_type & m, T const & v, std::vector<hsize_t> const &, std::vector<hsize_t> const & = std::vector<hsize_t>()) {
                    return m = &v; 
                }
                // TODO: es sollte ein set geben fuer T == U, das einen m-ptr zuruckgibt ...
                // TODO: make a nicer syntax for set(T & v, vec<size_t> s, vec<U> u, size_t o, size_t c)
                template<typename U> static typename boost::disable_if<typename boost::mpl::or_<
                    typename boost::is_same<U, std::complex<double> >::type,
                    typename boost::is_same<U, char *>::type
                >::type>::type set(T & v, std::vector<U> const & u, std::size_t o, std::vector<hsize_t> const & s, std::vector<hsize_t> const & c) {
                    if (s.size() != 1 || c.size() != 1 || c[0] == 0 || u.size() < o + c[0]) throw std::range_error("invalid data size");
                    std::copy(u.begin() + o, u.begin() + o + c[0], &v + s[0]);
                }
                template<typename U> static typename boost::enable_if<typename boost::mpl::or_<
                    typename boost::is_same<U, std::complex<double> >::type,
                    typename boost::is_same<U, char *>::type
                >::type>::type set(T &, std::vector<U> const &, std::size_t, std::vector<hsize_t> const &, std::vector<hsize_t> const &) { 
                    throw std::runtime_error("invalid type conversion"); 
                }
                static void resize(T &, std::vector<std::size_t> const &) {}
            };
            template<> struct matrix_type<std::string> : public boost::mpl::true_ {
                typedef std::string native_type;
                typedef char const * * pointer_type;
                typedef std::vector<char const *> buffer_type;
                typedef boost::mpl::true_ scalar;
                static std::vector<hsize_t> count(std::string const & v) { return size(v); }
                static std::vector<hsize_t> size(std::string const & v) { return std::vector<hsize_t>(1, 1); }
                static pointer_type get(buffer_type & m, std::string const & v, std::vector<hsize_t> const &, std::vector<hsize_t> const & = std::vector<hsize_t>()) {
                    m.resize(1);
                    return &(m[0] = v.c_str());
                }
                template<typename T> static typename boost::disable_if<typename boost::mpl::or_<
                    typename boost::is_same<T, std::complex<double> >::type,
                    typename boost::is_same<T, char *>::type
                >::type>::type set(std::string & v, std::vector<T> const & u, std::size_t o, std::vector<hsize_t> const & s, std::vector<hsize_t> const & c) { 
                    if (s.size() != 1 || c.size() != 1 || c[0] == 0 || u.size() < o + c[0]) throw std::range_error("invalid data size");
                    for (std::string * w = &v + s[0]; w != &v + s[0] + c[0]; ++w)
                        *w = boost::lexical_cast<std::string>(u[o - s[0] + (w - &v)]);
                }
                // TODO: string als skalar behandeln
                static void set(std::string & v, std::vector<char *> const & u, std::size_t o, std::vector<hsize_t> const & s, std::vector<hsize_t> const & c) { 
                    if (s.size() != 1 || c.size() != 1 || c[0] == 0 || u.size() < o + c[0]) throw std::range_error("invalid data size");
                    std::copy(u.begin() + o, u.begin() + o + c[0], &v);
                }
                static void set(std::string &, std::vector<std::complex<double> > const &, std::size_t, std::vector<hsize_t> const &, std::vector<hsize_t> const &) { 
                    throw std::runtime_error("invalid type conversion"); 
                }
                static void resize(std::string &, std::vector<std::size_t> const &) {}
            };
            template<typename T> struct matrix_type<std::complex<T> > : public boost::mpl::true_ {
                typedef std::complex<T> native_type;
                typedef internal_complex_type * pointer_type;
                typedef std::vector<internal_complex_type> buffer_type;
                typedef boost::mpl::true_ scalar;
                static std::vector<hsize_t> count(std::complex<T> const & v) { return size(v); }
                static std::vector<hsize_t> size(std::complex<T> const & v) { return std::vector<hsize_t>(1, 1); }
                static pointer_type get(buffer_type & m, std::complex<T> const & v, std::vector<hsize_t> const &, std::vector<hsize_t> const & t = std::vector<hsize_t>(1, 1)) {
                    if (t.size() != 1 || t[0] == 0) throw std::range_error("invalid data size");
                    m.resize(t[0]);
                    for (std::complex<T> const * u = &v; u != &v + t[0]; ++u) {
                        internal_complex_type c = { u->real(), u->imag() };
                        m[u - &v] = c;
                    }
                    return &m[0];
                }
                template<typename U> static typename boost::disable_if<
                    typename boost::is_same<U, std::complex<double> >::type
                >::type set(std::complex<T> &, std::vector<U> const &, std::size_t, std::vector<hsize_t> const &, std::vector<hsize_t> const &) { 
                    throw std::runtime_error("invalid type conversion"); 
                }
                static void set(std::complex<T> & v, std::vector<std::complex<double> > const & u, std::size_t o, std::vector<hsize_t> const & s, std::vector<hsize_t> const & c) { 
                    if (s.size() != 1 || c.size() != 1 || c[0] == 0 || u.size() < o + c[0]) throw std::range_error("invalid data size");
                    std::copy(u.begin() + o, u.begin() + o + c[0], &v + s[0]);
                }
                static void resize(std::complex<T> &, std::vector<std::size_t> const &) {}
            };
            template<> struct matrix_type<char const *> : public matrix_type<std::string> {
                typedef matrix_type<std::string>::pointer_type pointer_type;
                typedef matrix_type<std::string>::buffer_type buffer_type;
                template <typename T> static pointer_type get(buffer_type & m, T const & v, std::vector<hsize_t> const &, std::vector<hsize_t> const & = std::vector<hsize_t>()) {
                    m.resize(1);
                    return &(m[0] = v);
                }
                template<typename T, typename U> static void set(T &, U &, std::size_t, std::vector<hsize_t> const &, std::vector<hsize_t> const &) { 
                    throw std::runtime_error("no setter implemented"); 
                }
                template <typename T> static void resize(T &, std::vector<std::size_t> const &) {}
            };
            template<std::size_t N> struct matrix_type<char [N]> : public matrix_type<char const *> {};
            template<std::size_t N> struct matrix_type<char const [N]> : public matrix_type<char const *> {};
            #define HDF5_DEFINE_MATRIX_TYPE(C)                                                                                                             \
                template<typename T> struct matrix_type< C <T> > : public matrix_type<T> {                                                                 \
                    typedef typename matrix_type<T>::native_type native_type;                                                                              \
                    typedef typename matrix_type<T>::pointer_type pointer_type;                                                                            \
                    typedef typename matrix_type<T>::buffer_type buffer_type;                                                                              \
                    typedef boost::mpl::false_ scalar;                                                                                                     \
                    static std::vector<hsize_t> count( C <T> const & v) {                                                                                  \
                        if (v.size() == 0)                                                                                                                 \
                            return std::vector<hsize_t>(1, 0);                                                                                             \
                        else if (matrix_type<T>::scalar::value && boost::is_same<native_type, std::string>::value)                                         \
                            return std::vector<hsize_t>(1, 1);                                                                                             \
                        else if (matrix_type<T>::scalar::value)                                                                                            \
                            return size(v);                                                                                                                \
                        else {                                                                                                                             \
                            std::vector<hsize_t> c(1, 1), d(matrix_type<T>::count(v[0]));                                                                  \
                            std::copy(d.begin(), d.end(), std::back_inserter(c));                                                                          \
                            return c;                                                                                                                      \
                        }                                                                                                                                  \
                    }                                                                                                                                      \
                    static std::vector<hsize_t> size( C <T> const & v) {                                                                                   \
                        std::vector<hsize_t> s(1, v.size());                                                                                               \
                        if (!matrix_type<T>::scalar::value && v.size()) {                                                                                  \
                            std::vector<hsize_t> t(matrix_type<T>::size(v[0]));                                                                            \
                            std::copy(t.begin(), t.end(), std::back_inserter(s));                                                                          \
                            for (std::size_t i = 1; i < v.size(); ++i)                                                                                     \
                                if (!std::equal(t.begin(), t.end(), matrix_type<T>::size(v[i]).begin()))                                                   \
                                    throw std::range_error("no rectengual matrix");                                                                        \
                        }                                                                                                                                  \
                        return s;                                                                                                                          \
                    }                                                                                                                                      \
                    static pointer_type get(                                                                                                               \
                        buffer_type & m, C <T> const & v, std::vector<hsize_t> const & s, std::vector<hsize_t> const & = std::vector<hsize_t>()            \
                    ) {                                                                                                                                    \
                        if (matrix_type<T>::scalar::value)                                                                                                 \
                            return matrix_type<T>::get(m, v[s[0]], std::vector<hsize_t>(s.begin() + 1, s.end()), size(v));                                 \
                        else                                                                                                                               \
                            return matrix_type<T>::get(m, v[s[0]], std::vector<hsize_t>(s.begin() + 1, s.end()));                                          \
                    }                                                                                                                                      \
                    template<typename U> static void set(                                                                                                  \
                        C <T> & v, std::vector<U> const & u, std::size_t o, std::vector<hsize_t> const & s, std::vector<hsize_t> const & c                 \
                    ) {                                                                                                                                    \
                        if (matrix_type<T>::scalar::value)                                                                                                 \
                            matrix_type<T>::set(v[s[0]], u, o, s, c);                                                                                      \
                        else                                                                                                                               \
                            matrix_type<T>::set(v[s[0]], u, o, std::vector<hsize_t>(s.begin() + 1, s.end()), std::vector<hsize_t>(c.begin() + 1, c.end()));\
                    }                                                                                                                                      \
                    static void resize( C <T> & v, std::vector<std::size_t> const & s) {                                                                   \
                        if(                                                                                                                                \
                               !(s.size() == 1 && s[0] == 0)                                                                                               \
                            && ((matrix_type<T>::scalar::value && s.size() != 1) || (!matrix_type<T>::scalar::value && s.size() < 2))                      \
                        )                                                                                                                                  \
                            throw std::range_error("invalid data size");                                                                                   \
                        v.resize(s[0]);                                                                                                                    \
                        if (!matrix_type<T>::scalar::value)                                                                                                \
                            for (std::size_t i = 0; i < s[0]; ++i)                                                                                         \
                                matrix_type<T>::resize(v[i], std::vector<std::size_t>(s.begin() + 1, s.end()));                                            \
                    }                                                                                                                                      \
                };
            HDF5_DEFINE_MATRIX_TYPE(std::vector)
            HDF5_DEFINE_MATRIX_TYPE(std::valarray)
            HDF5_DEFINE_MATRIX_TYPE(boost::numeric::ublas::vector)
            template<typename T> struct matrix_type< std::pair<T *, std::vector<std::size_t> > > : public matrix_type<T> {
                typedef typename matrix_type<T>::native_type native_type;
                typedef typename matrix_type<T>::pointer_type pointer_type;
                typedef typename matrix_type<T>::buffer_type buffer_type;
                typedef boost::mpl::false_ scalar;
                static std::vector<hsize_t> count(std::pair<T *, std::vector<std::size_t> > const & v) {
                    if (matrix_type<T>::scalar::value && boost::is_same<native_type, std::string>::value)
                        return std::vector<hsize_t>(v.second.size(), 1);
                    else if (matrix_type<T>::scalar::value)
                        return std::vector<hsize_t>(v.second.begin(), v.second.end());
                    else {
                        std::vector<hsize_t> c(v.second.size(), 1), d(matrix_type<T>::count(*v.first));
                        std::copy(d.begin(), d.end(), std::back_inserter(c));
                        return c;
                    }
                }
                static std::vector<hsize_t> size(std::pair<T *, std::vector<std::size_t> > const & v) {
                    std::vector<hsize_t> s(v.second.begin(), v.second.end());
                    if (!matrix_type<T>::scalar::value && v.second.size()) {
                        std::vector<hsize_t> t(matrix_type<T>::size(*v.first));
                        std::copy(t.begin(), t.end(), std::back_inserter(s));
                        for (std::size_t i = 1; i < std::accumulate(s.begin(), s.end(), 1, std::multiplies<hsize_t>()); ++i)
                            if (!std::equal(t.begin(), t.end(), matrix_type<T>::size(*(v.first + i)).begin()))
                                throw std::range_error("no rectengual matrix");
                    }
                    return s;
                }
                static pointer_type get(buffer_type & m, std::pair<T *, std::vector<std::size_t> > const & v, std::vector<hsize_t> const & s, std::vector<hsize_t> const & = std::vector<hsize_t>()) {
                    if (matrix_type<T>::scalar::value)
                        return matrix_type<T>::get(
                              m
                            , *(v.first + std::accumulate(s.begin(), s.begin() + v.second.size(), 1, std::multiplies<hsize_t>()))
                            , std::vector<hsize_t>(s.begin() + v.second.size(), s.end())
                            , std::vector<hsize_t>(1, std::accumulate(v.second.begin(), v.second.end(), 1, std::multiplies<hsize_t>()))
                        );
                    else
                        return matrix_type<T>::get(
                              m
                            , *(v.first + std::accumulate(s.begin(), s.begin() + v.second.size(), 1, std::multiplies<hsize_t>()))
                            , std::vector<hsize_t>(s.begin() + v.second.size(), s.end())
                        );
                }
                template<typename U> static void set(std::pair<T *, std::vector<std::size_t> > & v, std::vector<U> const & u, std::size_t o, std::vector<hsize_t> const & s, std::vector<hsize_t> const & c) {
                    std::vector<hsize_t> start(1, std::accumulate(s.begin(), s.begin() + v.second.size(), 1, std::multiplies<hsize_t>()));
                    std::copy(s.begin() + v.second.size(), s.end(), std::back_inserter(start));
                    std::vector<hsize_t> count(1, std::accumulate(c.begin(), c.begin() + v.second.size(), 1, std::multiplies<hsize_t>()));
                    std::copy(c.begin() + v.second.size(), c.end(), std::back_inserter(count));
                    matrix_type<T>::set(*v.first, u, o, start, count);
                }
                static void resize(std::pair<T *, std::vector<std::size_t> > & v, std::vector<std::size_t> const & s) {
                    if (!(s.size() == 1 && s[0] == 0 && std::accumulate(v.second.begin(), v.second.end(), 0) == 0) && !std::equal(v.second.begin(), v.second.end(), s.begin()))
                        throw std::range_error("invalid data size");
                    if (!matrix_type<T>::scalar::value && s.size() > v.second.size())
                        for (std::size_t i = 0; i < std::accumulate(v.second.begin(), v.second.end(), 1, std::multiplies<hsize_t>()); ++i)
                            matrix_type<T>::resize(*(v.first + i), std::vector<std::size_t>(s.begin() + v.second.size(), s.end()));
                }
            };
            //TODO: boost::multiarray implementieren
            #undef HDF5_DEFINE_MATRIX_TYPE
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
            class error {
                public:
                    static herr_t noop(hid_t) { return 0; }
                    static herr_t callback(unsigned n, H5E_error2_t const * desc, void * buffer) {
                        *reinterpret_cast<std::ostringstream *>(buffer) << "    #" << n << " " << desc->file_name << " line " << desc->line << " in " << desc->func_name << "(): " << desc->desc << std::endl;
                        return 0;
                    }
                    static std::string invoke() {
                        std::ostringstream buffer;
                        buffer << "HDF5 error:" << std::endl;
                        H5Ewalk2(H5E_DEFAULT, H5E_WALK_DOWNWARD, callback, &buffer);
                        return buffer.str();
                    }
            };
            template<herr_t(*F)(hid_t)> class ressource {
                public:
                    ressource(): _id(-1) {}
                    ressource(hid_t id): _id(id) {
                        if (_id < 0)
                            throw std::runtime_error(error::invoke()); 
                        H5Eclear2(H5E_DEFAULT);
                    }
                    ~ressource() {
                        if(_id < 0 || F(_id) < 0) {
                            std::cerr << error::invoke() << std::endl;
                            std::abort();
                        }
                        H5Eclear2(H5E_DEFAULT); 
                    }
                    operator hid_t() const { 
                        return _id; 
                    }
                    ressource<F> & operator=(hid_t id) { 
                        if ((_id = id) < 0) 
                            throw std::runtime_error(error::invoke()); 
                        H5Eclear2(H5E_DEFAULT); 
                        return *this; 
                    }
                private:
                    hid_t _id;
            };
            typedef ressource<H5Fclose> file_type;
            typedef ressource<H5Gclose> group_type;
            typedef ressource<H5Dclose> data_type;
            typedef ressource<H5Aclose> attribute_type;
            typedef ressource<H5Sclose> space_type;
            typedef ressource<H5Tclose> type_type;
            typedef ressource<H5Pclose> property_type;
            typedef ressource<error::noop> error_type;
            template <typename T> T check_file(T id) { file_type unused(id); return unused; }
            template <typename T> T check_group(T id) { group_type unused(id); return unused; }
            template <typename T> T check_data(T id) { data_type unused(id); return unused; }
            template <typename T> T check_attribute(T id) { attribute_type unused(id); return unused; }
            template <typename T> T check_space(T id) { space_type unused(id); return unused; }
            template <typename T> T check_type(T id) { type_type unused(id); return unused; }
            template <typename T> T check_property(T id) { property_type unused(id); return unused; }
            template <typename T> T check_error(T id) { error_type unused(id); return unused; }
            template <typename Tag> class archive: boost::noncopyable {
                public:
                    struct log_type {
                        boost::posix_time::ptime time;
                        std::string name;
                    };
                    archive(std::string const & file): _revision(0), _state_id(-1), _log_id(-1), _filename(file) {
                        H5Eset_auto2(H5E_DEFAULT, NULL, NULL);
                        if (boost::is_same<Tag, write>::value) {
                            if (H5Fis_hdf5(file.c_str()) == 0)
                                throw std::runtime_error("no valid hdf5 file " + file);
                            hid_t id = H5Fopen(file.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
                            _file = (id < 0 ? H5Fcreate(file.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT) : id);
                            if (!is_group("/revisions")) {
                                set_group("/revisions");
                                set_attr("/revisions", "last", _revision);
                                internal_state_type::type v;
                                type_type state_id = H5Tenum_create(H5T_NATIVE_SHORT);
                                check_error(H5Tenum_insert(state_id, "CREATE", &(v = internal_state_type::CREATE)));
                                check_error(H5Tenum_insert(state_id, "PLACEHOLDER", &(v = internal_state_type::PLACEHOLDER)));
                                check_error(H5Tcommit2(_file, "state_type", state_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
                                type_type log_id = H5Tcreate (H5T_COMPOUND, sizeof(internal_log_type));
                                type_type time_id(H5Tcopy(H5T_C_S1));
                                check_error(H5Tset_size(time_id, H5T_VARIABLE));
                                check_error(H5Tinsert(log_id, "time", HOFFSET(internal_log_type, time), time_id));
                                type_type name_id(H5Tcopy(H5T_C_S1));
                                check_error(H5Tset_size(name_id, H5T_VARIABLE));
                                check_error(H5Tinsert(log_id, "log", HOFFSET(internal_log_type, name), name_id));
                                check_error(H5Tcommit2(_file, "log_type", log_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
                            }
                        } else {
                            if (check_error(H5Fis_hdf5(file.c_str())) == 0)
                                throw std::runtime_error("no valid hdf5 file " + file);
                            _file = H5Fopen(file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
                        }
                        _complex_id = H5Tcreate (H5T_COMPOUND, sizeof(internal_complex_type));
                        check_error(H5Tinsert(_complex_id, "r", HOFFSET(internal_complex_type, r), H5T_NATIVE_DOUBLE));
                        check_error(H5Tinsert(_complex_id, "i", HOFFSET(internal_complex_type, i), H5T_NATIVE_DOUBLE));
                        if (is_group("/revisions")) {
                            get_attr("/revisions", "last", _revision);
                            _log_id = check_error(H5Topen2(_file, "log_type", H5P_DEFAULT));
                            _state_id = check_error(H5Topen2(_file, "state_type", H5P_DEFAULT));
                        }
                    }
                    ~archive() {
                        H5Fflush(_file, H5F_SCOPE_GLOBAL);
                        if (_state_id > -1)
                            check_type(_state_id);
                        if (_log_id > -1)
                            check_type(_log_id);
                        if (
                               H5Fget_obj_count(_file, H5F_OBJ_DATATYPE) > (_state_id == -1 ? 0 : 1) + (_log_id == -1 ? 0 : 1)
                            || H5Fget_obj_count(_file, H5F_OBJ_ALL) - H5Fget_obj_count(_file, H5F_OBJ_FILE) - H5Fget_obj_count(_file, H5F_OBJ_DATATYPE) > 0
                        ) {
                            std::cerr << "Not all resources closed" << std::endl;
                            std::abort();
                        }
                    }
                    std::string const & filename() const {
                        return _filename;
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
                        set_attr("/revisions", "last", ++_revision);
                        set_group("/revisions/" + boost::lexical_cast<std::string>(_revision));
                        std::string time = boost::posix_time::to_iso_string(boost::posix_time::second_clock::local_time());
                        internal_log_type v = {
                            std::strcpy(new char[time.size() + 1], time.c_str()),
                            std::strcpy(new char[name.size() + 1], name.c_str())
                        };
                        set_attr("/revisions/" + boost::lexical_cast<std::string>(_revision), "info", v);
                        delete[] v.time;
                        delete[] v.name;
                    }
                    std::vector<std::pair<std::string, std::string> > list_revisions() const {
                        // TODO: implement
                        return std::vector<std::pair<std::string, std::size_t> >();
                    }
                    void export_revision(std::size_t revision, std::string const & file) const {
                        // TODO: implement
                    }
                    std::string get_context() const {
                        return _context;
                    }
                    void set_context(std::string const & context) {
                        _context = context;
                    }
                    std::string complete_path(std::string const & p) const {
                        if (p.size() && p[0] == '/')
                            return p;
                        else if (p.size() < 2 || p.substr(0, 2) != "..")
                            return _context + (_context.size() == 1 || !p.size() ? "" : "/") + p;
                        else {
                            std::string s = _context;
                            std::size_t i = 0;
                            for (; s.size() && p.substr(i, 2) == ".."; i += 3)
                                s = s.substr(0, s.find_last_of('/'));
                            return s + (s.size() == 1 || !p.substr(i).size() ? "" : "/") + p.substr(i);
                        }
                    }
                    template<typename T> typename boost::enable_if<
                        typename boost::mpl::and_<typename matrix_type<T>::type, typename boost::is_same<Tag, write>::type >
                    >::type serialize(std::string const & p, T const & v) {
                        if (p.find_last_of('@') != std::string::npos)
                            set_attr(complete_path(p).substr(0, complete_path(p).find_last_of('@') - 1), p.substr(p.find_last_of('@') + 1), v);
                        else
                            set_data(complete_path(p), v);
                    }
                    template<typename T> typename boost::enable_if<
                        typename boost::mpl::and_<typename matrix_type<T>::type, typename boost::is_same<Tag, read>::type >
                    >::type serialize(std::string const & p, T & v) {
                        if (p.find_last_of('@') != std::string::npos) {
                            #ifdef ALPS_HDF5_READ_GREEDY
                                if (is_attribute(p))
                            #endif
                                    get_attr(complete_path(p).substr(0, complete_path(p).find_last_of('@') - 1), p.substr(p.find_last_of('@') + 1), v);
                        } else
                            #ifdef ALPS_HDF5_READ_GREEDY
                                if (is_data(p))
                            #endif
                                    get_data(complete_path(p), v);
                    }
                    template<typename T> typename boost::disable_if<typename matrix_type<T>::type >::type serialize(std::string const & p, T & v) {
                        std::string c = get_context();
                        set_context(complete_path(p));
                        v.serialize(*this);
                        set_context(c);
                    }
                    void serialize(std::string const & p) {
                        if (p.find_last_of('@') != std::string::npos)
                            throw std::runtime_error("attributes needs to be a scalar type or a string" + p);
                        else
                            set_group(complete_path(p));
                    }
                    bool is_group(std::string const & p) const {
                        hid_t id = H5Gopen2(_file, complete_path(p).c_str(), H5P_DEFAULT);
                        return id < 0 ? false : check_group(id) != 0;
                    }
                    bool is_data(std::string const & p) const {
                        hid_t id = H5Dopen2(_file, complete_path(p).c_str(), H5P_DEFAULT);
                        return id < 0 ? false : check_data(id) != 0;
                    }
                    bool is_attribute(std::string const & p) const {
                        if (p.find_last_of('@') == std::string::npos)
                            throw std::runtime_error("no attribute paht: " + complete_path(p));
                        hid_t parent_id;
                        if (is_group(complete_path(p).substr(0, complete_path(p).find_last_of('@') - 1)))
                            parent_id = check_error(H5Gopen2(_file, complete_path(p).substr(0, complete_path(p).find_last_of('@') - 1).c_str(), H5P_DEFAULT));
                        else if (is_data(complete_path(p).substr(0, complete_path(p).find_last_of('@') - 1)))
                            parent_id = check_error(H5Dopen2(_file, complete_path(p).substr(0, complete_path(p).find_last_of('@') - 1).c_str(), H5P_DEFAULT));
                        else
                            #ifdef ALPS_HDF5_READ_GREEDY
                                return false;
                            #else
                                throw std::runtime_error("unknown path: " + complete_path(p));
                            #endif
                        bool exists = check_error(H5Aexists(parent_id, p.substr(p.find_last_of('@') + 1).c_str()));
                        if (is_group(complete_path(p).substr(0, complete_path(p).find_last_of('@') - 1)))
                            check_group(parent_id);
                        else
                            check_data(parent_id);
                        return exists;
                    }
                    std::vector<std::size_t> extent(std::string const & p) const {
                        if (is_null(p))
                            return std::vector<std::size_t>(1, 0);
                        else if (is_scalar(p))
                            return std::vector<std::size_t>(1, 1);
                        std::vector<hsize_t> buffer(dimensions(p), 0);
                        {
                            data_type data_id(H5Dopen2(_file, complete_path(p).c_str(), H5P_DEFAULT));
                            space_type space_id(H5Dget_space(data_id));
                            check_error(H5Sget_simple_extent_dims(space_id, &buffer.front(), NULL));
                        }
                        std::vector<std::size_t> extend(buffer.size(), 0);
                        std::copy(buffer.begin(), buffer.end(), extend.begin());
                        return extend;
                    }
                    std::size_t dimensions(std::string const & p) const {
                        data_type data_id(H5Dopen2(_file, complete_path(p).c_str(), H5P_DEFAULT));
                        space_type space_id(H5Dget_space(data_id));
                        return static_cast<hid_t>(check_error(H5Sget_simple_extent_dims(space_id, NULL, NULL)));
                    }
                    bool is_scalar(std::string const & p) const {
                        data_type data_id(H5Dopen2(_file, complete_path(p).c_str(), H5P_DEFAULT));
                        space_type space_id(H5Dget_space(data_id));
                        H5S_class_t type = H5Sget_simple_extent_type(space_id);
                        if (type == H5S_NO_CLASS)
                            throw std::runtime_error("error reading class " + complete_path(p));
                        return type == H5S_SCALAR;
                    }
                    bool is_null(std::string const & p) const {
                        data_type data_id(H5Dopen2(_file, complete_path(p).c_str(), H5P_DEFAULT));
                        space_type space_id(H5Dget_space(data_id));
                        H5S_class_t type = H5Sget_simple_extent_type(space_id);
                        if (type == H5S_NO_CLASS)
                            throw std::runtime_error("error reading class " + complete_path(p));
                        return type == H5S_NULL;
                    }
                    void delete_data(std::string const & p) {
                        if (is_data(p))
                            // TODO: implement provenance
                            check_error(H5Ldelete(_file, complete_path(p).c_str(), H5P_DEFAULT));
                        else
                            throw std::runtime_error("the path does not exists: " + p);
                    }
                    std::vector<std::string> list_children(std::string const & p) const {
                        std::vector<std::string> list;
                        group_type group_id(H5Gopen2(_file, complete_path(p).c_str(), H5P_DEFAULT));
                        check_error(H5Literate(group_id, H5_INDEX_NAME, H5_ITER_NATIVE, NULL, child_visitor, reinterpret_cast<void *>(&list)));
                        return list;
                    }
                    std::vector<std::string> list_attr(std::string const & p) const {
                        std::vector<std::string> list;
                        if (is_group(p)) {
                            group_type id(H5Gopen2(_file, complete_path(p).c_str(), H5P_DEFAULT));
                            check_error(H5Aiterate2(id, H5_INDEX_CRT_ORDER, H5_ITER_NATIVE, NULL, attr_visitor, reinterpret_cast<void *>(&list)));
                        } else {
                            data_type id(H5Dopen2(_file, complete_path(p).c_str(), H5P_DEFAULT));
                            check_error(H5Aiterate2(id, H5_INDEX_CRT_ORDER, H5_ITER_NATIVE, NULL, attr_visitor, reinterpret_cast<void *>(&list)));
                        }
                        return list;
                    }
                private:
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
                    template<typename T> hid_t get_native_type(std::complex<T>) const { return H5Tcopy(_complex_id); }
                    hid_t get_native_type(std::string) const { 
                        hid_t type_id = H5Tcopy(H5T_C_S1);
                        check_error(H5Tset_size(type_id, H5T_VARIABLE));
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
                        hid_t data_id = H5Dopen2(_file, p.c_str(), H5P_DEFAULT), tmp_id;
                        if (data_id < 0) {
                            if (p.find_last_of('/') < std::string::npos && p.find_last_of('/') > 0)
                                set_group(p.substr(0, p.find_last_of('/')));
                            data_id = create_dataset(p, type_id, space_id, d, s, set_prop);
                        } else if (
                               (d > 0 && s[0] > 0 && is_null(p)) 
                            || (d > 0 && s[0] == 0 && !is_null(p)) 
                            || !check_error(H5Tequal(type_type(H5Dget_type(data_id)), type_type(H5Tcopy(type_id))))
                            || (d > 0 && s[0] > 0 && H5Dset_extent(data_id, s) < 0)
                        ) {
                            std::vector<std::string> names = list_attr(p);
                            if (names.size()) {
                                tmp_id = H5Gcreate2(_file, "/revisions/waitingroom", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                                copy_attributes(tmp_id, data_id, names);
                            }
                            check_data(data_id);
                            check_error(H5Ldelete(_file, p.c_str(), H5P_DEFAULT));
                            data_id = create_dataset(p, type_id, space_id, d, s, set_prop);
                            if (names.size()) {
                                copy_attributes(data_id, tmp_id, names);
                                check_group(tmp_id);
                                check_error(H5Ldelete(_file, "/revisions/waitingroom", H5P_DEFAULT));
                            }
                        }
                        return data_id;
                    }
                    hid_t create_dataset(std::string const & p, hid_t type_id, hid_t space_id, hsize_t d, hsize_t const * s = NULL, bool set_prop = true) const {
                        if (set_prop) {
                            property_type prop_id(H5Pcreate(H5P_DATASET_CREATE));
                            check_error(H5Pset_fill_time(prop_id, H5D_FILL_TIME_NEVER));
                            if (d > 0)
                                check_error(H5Pset_chunk(prop_id, d, s));
                            return H5Dcreate2(_file, p.c_str(), type_id, space_type(space_id), H5P_DEFAULT, prop_id, H5P_DEFAULT);
                        } else
                            return H5Dcreate2(_file, p.c_str(), type_id, space_type(space_id), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                    }
                    void copy_attributes(hid_t dest_id, hid_t source_id, std::vector<std::string> const & names) const {
                        for (std::vector<std::string>::const_iterator it = names.begin(); it != names.end(); ++it) {
                            attribute_type attr_id = H5Aopen(source_id, it->c_str(), H5P_DEFAULT);
                            type_type type_id = H5Aget_type(attr_id);
                            if (H5Tget_class(type_id) == H5T_STRING) {
                                std::string v;
                                v.resize(H5Tget_size(type_id));
                                check_error(H5Aread(attr_id, type_type(H5Tcopy(type_id)), &v[0]));
                                attribute_type new_id = H5Acreate2(dest_id, it->c_str(), type_id, space_type(H5Screate(H5S_SCALAR)), H5P_DEFAULT, H5P_DEFAULT);
                                check_error(H5Awrite(new_id, type_id, &v[0]));
                            } else if (check_error(H5Tequal(type_type(H5Tcopy(type_id)), type_type(H5Tcopy(_state_id)))) > 0) {
                                internal_state_type::type v;
                                check_error(H5Aread(attr_id, _state_id, &v));
                                attribute_type new_id = H5Acreate2(dest_id, it->c_str(), _state_id, space_type(H5Screate(H5S_SCALAR)), H5P_DEFAULT, H5P_DEFAULT);
                                check_error(H5Awrite(new_id, _state_id, &v));
                            }
                            #define HDF5_COPY_ATTR(T)                                                                                                      \
                                else if (check_error(H5Tequal(type_type(H5Tcopy(type_id)), type_type(get_native_type(static_cast<T>(0))))) > 0) {          \
                                    T v;                                                                                                                   \
                                    check_error(H5Aread(attr_id, type_type(H5Tcopy(type_id)), &v));                                                        \
                                    attribute_type new_id = H5Acreate2(                                                                                    \
                                        dest_id, it->c_str(), type_id, space_type(H5Screate(H5S_SCALAR)), H5P_DEFAULT, H5P_DEFAULT                         \
                                    );                                                                                                                     \
                                    check_error(H5Awrite(new_id, type_id, &v));                                                                            \
                                }
                            HDF5_FOREACH_SCALAR(HDF5_COPY_ATTR)
                            #undef HDF5_COPY_ATTR
                            else throw std::runtime_error("error in copying attribute: " + *it);
                        }
                    }
                    hid_t save_comitted_data(std::string const & p, hid_t type_id, hid_t space_id, hsize_t d, hsize_t const * s = NULL, bool set_prop = true) const {
                        std::string rev_path = "/revisions/" + boost::lexical_cast<std::string>(_revision) + p;
                        if (_revision && !is_data(p))
                            set_data(rev_path, internal_state_type::CREATE);
                        else if (_revision) {
                            hid_t data_id = H5Dopen2(_file, rev_path.c_str(), H5P_DEFAULT);
                            std::vector<std::string> revision_names;
                            if (data_id > 0 && check_error(H5Tequal(type_type(H5Dget_type(data_id)), type_type(H5Tcopy(_state_id)))) > 0) {
                                internal_state_type::type v;
                                check_error(H5Dread(data_id, _state_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, &v));
                                if (v == internal_state_type::PLACEHOLDER) {
                                    if ((revision_names = list_attr(rev_path)).size()) {
                                        group_type tmp_id = H5Gcreate2(_file, "/revisions/waitingroom", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                                        copy_attributes(tmp_id, data_id, revision_names);
                                    }
                                    check_data(data_id);
                                    check_error(H5Ldelete(_file, rev_path.c_str(), H5P_DEFAULT));
                                } else
                                    check_data(data_id);
                            } else if (data_id >= 0)
                                check_data(data_id);
                            if (!is_data(rev_path)) {
                                set_group(rev_path.substr(0, rev_path.find_last_of('/')));
                                check_error(H5Lmove(_file, p.c_str(), H5L_SAME_LOC, (rev_path).c_str(), H5P_DEFAULT, H5P_DEFAULT));
                                hid_t new_id = create_path(p, type_id, space_id, d, s, set_prop);
                                std::vector<std::string> current_names = list_attr(rev_path);
                                data_type data_id(H5Dopen2(_file, rev_path.c_str(), H5P_DEFAULT));
                                copy_attributes(new_id, data_id, current_names); 
                                for (std::vector<std::string>::const_iterator it = current_names.begin(); it != current_names.end(); ++it)
                                    H5Adelete(data_id, it->c_str());
                                if (revision_names.size()) {
                                    copy_attributes(data_id, group_type(H5Gopen2(_file, "/revisions/waitingroom", H5P_DEFAULT)), revision_names);
                                    check_error(H5Ldelete(_file, "/revisions/waitingroom", H5P_DEFAULT));
                                }
                                return new_id;
                            }
                        }
                        return create_path(p, type_id, space_id, d, s, set_prop);
                    }
                    // TODO: optimize for T == U
                    template<typename T, typename U> void get_helper(T & v, hid_t data_id, hid_t type_id, bool is_attr) const {
                       std::vector<hsize_t> size(matrix_type<T>::size(v)), start(size.size(), 0), count(matrix_type<T>::count(v));
                       std::vector<U> data(std::accumulate(size.begin(), size.end(), 1, std::multiplies<std::size_t>()));
                       if (is_attr)
                           check_error(H5Aread(data_id, type_id, &data.front()));
                       else
                           check_error(H5Dread(data_id, type_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, &data.front()));
                        if (std::equal(count.begin(), count.end(), size.begin()))
                            matrix_type<T>::set(v, data, 0, start, count);
                        else {
                            std::size_t last = count.size() - 1, pos;
                            for(;count[last] == size[last]; --last);
                            do {
                                std::size_t offset = 0;
                                for (std::size_t i = start.size(), sum = 1; i > 0; --i) {
                                    offset += sum * start[i - 1];
                                    sum *= size[i - 1];
                                }
                                matrix_type<T>::set(v, data, offset, start, count);
                                if (++start[last] == size[last] && last) {
                                    for (pos = last; pos && start[pos] == size[pos]; --pos);
                                    ++start[pos];
                                    for (++pos; pos <= last; ++pos)
                                        start[pos] = 0;
                                }
                            } while (start[0] < size[0]);
                        }
                        if (boost::is_same<T, char *>::value)
                            check_error(H5Dvlen_reclaim(type_id, space_type(H5Dget_space(data_id)), H5P_DEFAULT, &data.front()));
                    }
                   template<typename T> void get_data(std::string const & p, T & v) const {
                       if (is_scalar(p) != matrix_type<T>::scalar::value)
                           throw std::runtime_error("scalar - vector conflict");
                       else if (matrix_type<T>::scalar::value && is_null(p))
                           throw std::runtime_error("scalars cannot be null");
                       else if (is_null(p))
                           matrix_type<T>::resize(v, std::vector<std::size_t>(1, 0));
                       else {
                           std::vector<hsize_t> size(dimensions(p), 0);
                           data_type data_id(H5Dopen2(_file, p.c_str(), H5P_DEFAULT));
                           type_type type_id(H5Dget_type(data_id));
                           type_type native_id(H5Tget_native_type(type_id, H5T_DIR_ASCEND));
                           if (size.size()) {
                               space_type space_id(H5Dget_space(data_id));
                               check_error(H5Sget_simple_extent_dims(space_id, &size.front(), NULL));
                           }
                           matrix_type<T>::resize(v, std::vector<std::size_t>(size.begin(), size.end()));
                           if (H5Tget_class(native_id) == H5T_STRING)
                               get_helper<T, char *>(v, data_id, type_id, false);
                           else if (check_error(H5Tequal(type_type(H5Tcopy(_complex_id)), type_type(H5Tcopy(type_id)))))
                               get_helper<T, std::complex<double> >(v, data_id, type_id, false);
                           #define HDF5_GET_STRING(U)                                                                                                      \
                               else if (check_error(H5Tequal(type_type(H5Tcopy(native_id)), type_type(get_native_type(static_cast<U>(0))))) > 0)           \
                                   get_helper<T, U>(v, data_id, type_id, false);
                           HDF5_FOREACH_SCALAR(HDF5_GET_STRING)
                           #undef HDF5_GET_STRING
                           else throw std::runtime_error("invalid type");
                       }
                   }
                   template<typename T> void get_attr(std::string const & p, std::string const & s, T & v) const {
                       hid_t parent_id;
                       if (!matrix_type<T>::scalar::value)
                           throw std::runtime_error("attributes need to be scalar");
                       else if (is_group(p))
                           parent_id = H5Gopen2(_file, p.c_str(), H5P_DEFAULT);
                       else if (is_data(p))
                           parent_id = H5Dopen2(_file, p.c_str(), H5P_DEFAULT);
                       else
                           throw std::runtime_error("invalid path");
                       attribute_type attr_id(H5Aopen(parent_id, s.c_str(), H5P_DEFAULT));
                       type_type type_id(H5Aget_type(attr_id));
                       type_type native_id(H5Tget_native_type(type_id, H5T_DIR_ASCEND));
                       if (H5Tget_class(native_id) == H5T_STRING)
                           get_helper<T, char *>(v, attr_id, type_id, true);
                       else if (check_error(H5Tequal(type_type(H5Tcopy(_complex_id)), type_type(H5Tcopy(type_id)))))
                           get_helper<T, std::complex<double> >(v, attr_id, type_id, true);
                       #define HDF5_GET_ATTR(U)                                                                                                            \
                           else if (check_error(H5Tequal(type_type(H5Tcopy(native_id)), type_type(get_native_type(static_cast<U>(0))))) > 0)               \
                               get_helper<T, U>(v, attr_id, type_id, true);
                       HDF5_FOREACH_SCALAR(HDF5_GET_ATTR)
                       #undef HDF5_GET_ATTR
                       else throw std::runtime_error("invalid type");
                       if (is_group(p))
                           check_group(parent_id);
                       else
                           check_data(parent_id);
                   }
                  template<typename T> void set_data(std::string const & p, T const & v) const {
                        type_type type_id(get_native_type(typename matrix_type<T>::native_type()));
                        std::vector<hsize_t> size(matrix_type<T>::size(v)), start(size.size(), 0), count(matrix_type<T>::count(v));
                        typename matrix_type<T>::buffer_type mem;
                        if (matrix_type<T>::scalar::value) {
                            data_type data_id(save_comitted_data(p, type_id, H5Screate(H5S_SCALAR), 0, NULL, !boost::is_same<typename matrix_type<T>::native_type, std::string>::value));
                            check_error(H5Dwrite(data_id, type_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, matrix_type<T>::get(mem, v, start)));
                        } else if (std::accumulate(size.begin(), size.end(), 0) == 0)
                            check_data(save_comitted_data(p, type_id, H5Screate(H5S_NULL), 0, NULL, !boost::is_same<typename matrix_type<T>::native_type, std::string>::value));
                        else {
                            data_type data_id(save_comitted_data(p, type_id, H5Screate_simple(size.size(), &size.front(), NULL), size.size(), &size.front(), !boost::is_same<typename matrix_type<T>::native_type, std::string>::value));
                            if (std::equal(count.begin(), count.end(), size.begin()))
                                check_error(H5Dwrite(data_id, type_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, matrix_type<T>::get(mem, v, start)));
                            else {
                                std::size_t last = count.size() - 1, pos;
                                for(;count[last] == size[last]; --last);
                                do {
                                    space_type space_id(H5Dget_space(data_id));
                                    check_error(H5Sselect_hyperslab(space_id, H5S_SELECT_SET, &start.front(), NULL, &count.front(), NULL));
                                    space_type mem_id(H5Screate_simple(count.size(), &count.front(), NULL));
                                    check_error(H5Dwrite(data_id, type_id, mem_id, space_id, H5P_DEFAULT, matrix_type<T>::get(mem, v, start)));
                                    if (++start[last] == size[last] && last) {
                                        for (pos = last; pos && start[pos] == size[pos]; --pos);
                                        ++start[pos];
                                        for (++pos; pos <= last; ++pos)
                                            start[pos] = 0;
                                    }
                                } while (start[0] < size[0]);
                            }
                        }
                    }
                    template<typename T> void set_attr(std::string const & p, std::string const & s, T const & v) const {
                        hid_t parent_id;
                        std::string rev_path = "/revisions/" + boost::lexical_cast<std::string>(_revision) + p;
                        if (!matrix_type<T>::scalar::value)
                            throw std::runtime_error("attributes need to be scalar");
                        else if (is_group(p)) {
                            parent_id = check_error(H5Gopen2(_file, p.c_str(), H5P_DEFAULT));
                            if (_revision && p.substr(0, std::strlen("/revisions")) != "/revisions" && !is_group(rev_path))
                                set_group(rev_path);
                        } else if (is_data(p)) {
                            parent_id = check_error(H5Dopen2(_file, p.c_str(), H5P_DEFAULT));
                            if (_revision && p.substr(0, std::strlen("/revisions")) != "/revisions" && !is_data(rev_path))
                                set_data(rev_path, internal_state_type::PLACEHOLDER);
                        } else
                            throw std::runtime_error("unknown path: " + p);
                        if (_revision && p.substr(0, std::strlen("/revisions")) != "/revisions" && !check_error(H5Aexists(parent_id, s.c_str())))
                            set_attr(rev_path, s, internal_state_type::CREATE);
                        else if (_revision && p.substr(0, std::strlen("/revisions")) != "/revisions") {
                            hid_t data_id = (is_group(rev_path) ? H5Gopen2(_file, rev_path.c_str(), H5P_DEFAULT) : H5Dopen2(_file, rev_path.c_str(), H5P_DEFAULT));
                            if (check_error(H5Aexists(data_id, s.c_str())) && check_error(H5Tequal(type_type(H5Aget_type(attribute_type(H5Aopen(data_id, s.c_str(), H5P_DEFAULT)))), type_type(H5Tcopy(_state_id)))) > 0)
                                H5Adelete(data_id, s.c_str());
                            if (!check_error(H5Aexists(data_id, s.c_str())))
                                copy_attributes(data_id, parent_id, std::vector<std::string>(1, s));
                            if (is_group(p))
                                check_group(data_id);
                            else
                                check_data(data_id);
                        }
                        hid_t id = H5Aopen(parent_id, s.c_str(), H5P_DEFAULT);
                        type_type type_id(get_native_type(typename matrix_type<T>::native_type()));
                        if (id >= 0 && check_error(H5Tequal(type_type(H5Aget_type(id)), type_type(H5Tcopy(type_id)))) == 0) {
                            check_attribute(id);
                            H5Adelete(parent_id, s.c_str());
                            id = -1;
                        }
                        if (id < 0)
                            id = H5Acreate2(parent_id, s.c_str(), type_id, space_type(H5Screate(H5S_SCALAR)), H5P_DEFAULT, H5P_DEFAULT);
                        attribute_type attr_id(id);
                        typename matrix_type<T>::buffer_type mem;
                        check_error(H5Awrite(attr_id, type_id, matrix_type<T>::get(mem, v, std::vector<hsize_t>(1, 1))));
                        if (is_group(p))
                            check_group(parent_id);
                        else
                            check_data(parent_id);
                    }
                    void set_group(std::string const & p) const {
                        if (!is_group(p)) {
                            std::size_t pos;
                            hid_t group_id = -1;
                            for (pos = p.find_last_of('/'); group_id < 0 && pos > 0 && pos < std::string::npos; pos = p.find_last_of('/', pos - 1))
                                group_id = H5Gopen2(_file, p.substr(0, pos).c_str(), H5P_DEFAULT);
                            if (group_id < 0) {
                                if ((pos = p.find_first_of('/', 1)) != std::string::npos)
                                    check_group(H5Gcreate2(_file, p.substr(0, pos).c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
                            } else {
                                pos = p.find_first_of('/', pos + 1);
                                check_group(group_id);
                            }
                            while (pos != std::string::npos && (pos = p.find_first_of('/', pos + 1)) != std::string::npos && pos > 0)
                                check_group(H5Gcreate2(_file, p.substr(0, pos).c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
                            check_group(H5Gcreate2(_file, p.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
                        }
                    }
                    int _revision;
                    hid_t _state_id;
                    hid_t _log_id;
                    hid_t _complex_id;
                    std::string _context;
                    std::string _filename;
                    file_type _file;
            };
        }
        typedef detail::archive<detail::read> iarchive;
        typedef detail::archive<detail::write> oarchive;
        template <typename T> iarchive & serialize(iarchive & ar, std::string const & p, T & v) {
            ar.serialize(p, v);
            return ar;
        }
        template <typename T> oarchive & serialize(oarchive & ar, std::string const & p, T const & v) {
            ar.serialize(p, v);
            return ar;
        }
        namespace detail {
            template <typename T, bool B> class pvp;
            template <typename T, bool B> archive<write> & operator<< (archive<write> & ar, pvp<T, B> const & v) { return v.serialize(ar); }
            template <typename T, bool B> archive<read> & operator>> (archive<read> & ar, pvp<T, B> const & v) { return v.serialize(ar); }
        }
    }
    #define HDF5_MAKE_PVP(ref_type)                                                                                                                        \
        template <typename T> hdf5::detail::pvp<T ref_type, hdf5::detail::matrix_type<T>::value> make_pvp(std::string const & p, T ref_type v) {           \
            return hdf5::detail::pvp<T ref_type, hdf5::detail::matrix_type<T>::value>(p, v);                                                               \
        }                                                                                                                                                  \
        template <typename T> hdf5::detail::pvp<T ref_type, hdf5::detail::matrix_type<T>::value> make_pvp(std::string const & p, boost::shared_ptr<T> ref_type v) { \
            return hdf5::detail::pvp<T ref_type, hdf5::detail::matrix_type<T>::value>(p, *v);                                                              \
        }                                                                                                                                                  \
        template <typename T> hdf5::detail::pvp<T ref_type, hdf5::detail::matrix_type<T>::value> make_pvp(std::string const & p, std::auto_ptr<T> ref_type v) { \
            return hdf5::detail::pvp<T ref_type, hdf5::detail::matrix_type<T>::value>(p, *v);                                                              \
        }                                                                                                                                                  \
        template <typename T> hdf5::detail::pvp<T ref_type, hdf5::detail::matrix_type<T>::value> make_pvp(std::string const & p, boost::weak_ptr<T> ref_type v) { \
            return hdf5::detail::pvp<T ref_type, hdf5::detail::matrix_type<T>::value>(p, *v);                                                              \
        }                                                                                                                                                  \
        template <typename T> hdf5::detail::pvp<T ref_type, hdf5::detail::matrix_type<T>::value> make_pvp(std::string const & p, boost::intrusive_ptr<T> ref_type v) { \
            return hdf5::detail::pvp<T ref_type, hdf5::detail::matrix_type<T>::value>(p, *v);                                                              \
        }                                                                                                                                                  \
        template <typename T> hdf5::detail::pvp<T ref_type, hdf5::detail::matrix_type<T>::value> make_pvp(std::string const & p, boost::scoped_ptr<T> ref_type v) { \
            return hdf5::detail::pvp<T ref_type, hdf5::detail::matrix_type<T>::value>(p, *v);                                                              \
        }
    HDF5_MAKE_PVP(&)
    HDF5_MAKE_PVP(const &)
    #undef HDF5_MAKE_PVP
    template <typename T> hdf5::detail::pvp<std::pair<T *, std::vector<std::size_t> >, hdf5::detail::matrix_type<T>::value> make_pvp(std::string const & p, T * v, std::size_t s) {
        return hdf5::detail::pvp<std::pair<T *, std::vector<std::size_t> >, hdf5::detail::matrix_type<T>::value>(p, std::make_pair(v, std::vector<std::size_t>(1, s)));
    }
    template <typename T> hdf5::detail::pvp<std::pair<T const *, std::vector<std::size_t> >, hdf5::detail::matrix_type<T>::value> make_pvp(std::string const & p, T const * v, std::size_t s) {
        return hdf5::detail::pvp<std::pair<T const *, std::vector<std::size_t> >, hdf5::detail::matrix_type<T>::value>(p, std::make_pair(v, std::vector<std::size_t>(1, s)));
    }
    template <typename T> hdf5::detail::pvp<std::pair<T *, std::vector<std::size_t> >, hdf5::detail::matrix_type<T>::value> make_pvp(std::string const & p, T * v, std::vector<std::size_t> const & s) {
        return hdf5::detail::pvp<std::pair<T *, std::vector<std::size_t> >, hdf5::detail::matrix_type<T>::value>(p, std::make_pair(v, s));
    }
    template <typename T> hdf5::detail::pvp<std::pair<T const *, std::vector<std::size_t> >, hdf5::detail::matrix_type<T>::value> make_pvp(std::string const & p, T const * v, std::vector<std::size_t> const & s) {
        return hdf5::detail::pvp<std::pair<T const *, std::vector<std::size_t> >, hdf5::detail::matrix_type<T>::value>(p, std::make_pair(v, s));
    }
    namespace hdf5 {
        namespace detail {
            template <typename T, bool B> class pvp {
                public:
                    pvp(std::string const & p, T v): _p(p), _v(v) {}
                    pvp(pvp<T, B> const & c): _p(c._p), _v(c._v) {}
                    template<typename Tag> archive<Tag> & serialize(archive<Tag> & ar) const { return ::alps::hdf5::serialize(ar, _p, _v); }
                private:
                    std::string _p;
                    mutable T _v;
            };
            // TODO: wenn man den generic_type als Skalar behandelt, dann sollte man eigentlich mit der normalen architektur durchkommen ....
            #define HDF5_DEFINE_GENERIC_TYPE(C, ref_type)                                                                                                  \
                template <typename T> class pvp < C <T> ref_type, false> {                                                                                 \
                    public:                                                                                                                                \
                        pvp(std::string const & p, C <T> ref_type v): _p(p), _v(v) {}                                                                      \
                        pvp(pvp<C <T>, false> const & c): _p(c._p), _v(c._v) {}                                                                            \
                        ::alps::hdf5::iarchive & serialize(::alps::hdf5::iarchive & ar) const {                                                            \
                            std::vector<std::string> children = ar.list_children(ar.complete_path(_p));                                                    \
                            _v.resize(children.size());                                                                                                    \
                            for (std::vector<std::string>::const_iterator it = children.begin(); it != children.end(); ++it)                               \
                                ar >> ::alps::make_pvp(ar.complete_path(_p) + "/" + *it, _v[it - children.begin()]);                                       \
 		                    return ar;                                                                                                                     \
                       }                                                                                                                                   \
                        ::alps::hdf5::oarchive & serialize(::alps::hdf5::oarchive & ar) const {                                                            \
                            if (!_v.size())                                                                                                                \
                                ar << ::alps::make_pvp(ar.complete_path(_p), std::vector<int>(0));                                                         \
                            else                                                                                                                           \
                                for (std::size_t i = 0; i < _v.size(); ++i)                                                                                \
                                    ar << ::alps::make_pvp(ar.complete_path(_p) + "/" + boost::lexical_cast<std::string>(i), _v[i]);                       \
		                    return ar;                                                                                                                     \
                        }                                                                                                                                  \
                    private:                                                                                                                               \
                        std::string _p;                                                                                                                    \
                        mutable C <T> ref_type _v;                                                                                                         \
                };
            #define HDF5_DEFINE_GENERIC_TYPE_CONST(ref_type)                                                                                               \
                HDF5_DEFINE_GENERIC_TYPE(std::vector, ref_type)                                                                                            \
                HDF5_DEFINE_GENERIC_TYPE(std::valarray, ref_type)                                                                                          \
                HDF5_DEFINE_GENERIC_TYPE(boost::numeric::ublas::vector, ref_type)
            HDF5_DEFINE_GENERIC_TYPE_CONST(&)
            HDF5_DEFINE_GENERIC_TYPE_CONST(const &)
            #undef HDF5_DEFINE_GENERIC_TYPE_CONST
            #undef HDF5_DEFINE_GENERIC_TYPE
            #undef HDF5_FOREACH_SCALAR
        }
    }
}


#endif

#endif
