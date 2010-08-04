/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2010 by Lukas Gamper <gamperl -at- gmail.com>
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

#include <string>
#include <vector>
#include <complex>
#include <valarray>
#include <iostream>
#include <algorithm>

#include <boost/random.hpp>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>

#include <alps/hdf5.hpp>

typedef enum { PLUS, MINUS } enum_type;
template<typename T> struct creator;
template<typename T> class userdefined_class;
template<typename T, typename U> class cast_type;

double rng() {
    static boost::mt19937 rng(SEED);
    static boost::uniform_real<> dist_real(0.,1e12);
    static boost::variate_generator<boost::mt19937, boost::uniform_real<> > random_real(rng, dist_real);
    return random_real();
}

template<typename T> void initialize(T & v) {
    v = static_cast<T>(rng());
}
template<typename T> void initialize(std::complex<T> & v) {
    v = std::complex<T>(rng(), rng());
}
void initialize(std::string & v) {
    v = boost::lexical_cast<std::string>(rng());
}
void initialize(enum_type & v) {
    v = static_cast<std::size_t>(rng()) % 2 == 0 ? PLUS : MINUS;
}
template<typename T> void initialize(userdefined_class<T> & v) {}

template<typename T, typename U> void initialize(cast_type<T, U> & v) {}

template<typename T> bool equal(T const & a, T const & b);

template<typename T> class userdefined_class {
    public:
        userdefined_class(): b(VECTOR_SIZE) {
            initialize(a);
            for (std::size_t i = 0; i < VECTOR_SIZE; ++i)
                initialize(b[i]);
            initialize(c);
        }
        void serialize(alps::hdf5::iarchive & ar) {
            ar
                >> alps::make_pvp("a", a)
                >> alps::make_pvp("b", b)
                >> alps::make_pvp("c", c)
            ;
        }
        void serialize(alps::hdf5::oarchive & ar) const { 
            ar
                << alps::make_pvp("a", a)
                << alps::make_pvp("b", b)
                << alps::make_pvp("c", c)
            ;
        }
        bool operator==(userdefined_class<T> const & v) const {
            return a == v.a && b.size() == v.b.size() && std::equal(b.begin(), b.end(), v.b.begin()) && c == v.c;
        }
    private:
        T a;
        std::vector<T> b;
        enum_type c;
};

template<typename T, typename U> class cast_type_base {
    public:
        cast_type_base(T const & v = T()): has_u(false), t(v) {}
        void serialize(alps::hdf5::iarchive & ar) {
            has_u = true;
            ar
                >> alps::make_pvp("t", u)
            ;
        }
        void serialize(alps::hdf5::oarchive & ar) const { 
            ar
                << alps::make_pvp("t", t)
            ;
        }
    protected:
        bool has_u;
        T t;
        U u;
};

template<typename T, typename U> class cast_type : public cast_type_base<T, U> {
    public:
        typedef cast_type_base<T, U> base_type;
        cast_type(): base_type() {
            initialize(base_type::t);
        }
        bool operator==(cast_type<T, U> const & v) const {
            return (base_type::has_u ? base_type::u : boost::lexical_cast<U>(base_type::t)) == (v.has_u ? v.u : boost::lexical_cast<U>(v.t));
        }
};

#define HDF5_DEFINE_VECTOR_VECTOR_CAST_TYPE(C, D)                                                  \
template<typename T, typename U> class cast_type< C <T>, D <U> >                                   \
    : public cast_type_base< C <T>, D <U> >                                                        \
{                                                                                                  \
    public:                                                                                        \
        typedef cast_type_base<C <T>, D <U> > base_type;                                           \
        cast_type(): base_type(creator< C <T> >::random()) {}                                      \
        bool operator==(cast_type< C <T>, D <U> > const & v) const {                               \
            if (base_type::has_u && !v.has_u)                                                      \
                return std::equal(&base_type::u[0], &base_type::u[0] + base_type::u.size(), &v.t[0]);\
            else if (!base_type::has_u && v.has_u)                                                 \
                return std::equal(&base_type::t[0], &base_type::t[0] + base_type::t.size(), &v.u[0]);\
            else                                                                                   \
                return false;                                                                      \
        }                                                                                          \
};
HDF5_DEFINE_VECTOR_VECTOR_CAST_TYPE(std::valarray, std::vector)
HDF5_DEFINE_VECTOR_VECTOR_CAST_TYPE(std::vector, std::valarray)
HDF5_DEFINE_VECTOR_VECTOR_CAST_TYPE(std::valarray, boost::numeric::ublas::vector)
HDF5_DEFINE_VECTOR_VECTOR_CAST_TYPE(boost::numeric::ublas::vector, std::valarray)
HDF5_DEFINE_VECTOR_VECTOR_CAST_TYPE(boost::numeric::ublas::vector, std::vector)
HDF5_DEFINE_VECTOR_VECTOR_CAST_TYPE(std::vector, boost::numeric::ublas::vector)
#undef HDF5_DEFINE_VECTOR_VECTOR_CAST_TYPE

template<typename T, typename U> class cast_type< std::pair<T *, std::vector<std::size_t> >, std::vector<std::vector<std::vector<U> > > >
    : public cast_type_base< std::pair<T *, std::vector<std::size_t> >, std::vector<std::vector<std::vector<U> > > >
{
    public:
        typedef cast_type_base< std::pair<T *, std::vector<std::size_t> >, std::vector<std::vector<std::vector<U> > > > base_type;
        cast_type(): base_type(creator<std::pair<T *, std::vector<std::size_t> > >::special()) {}
        bool operator==(cast_type<std::pair<T *, std::vector<std::size_t> >, std::vector<std::vector<std::vector<U> > > > const & v) const {
            if (base_type::has_u == v.has_u)
                return false;
            if (base_type::has_u)
                return v == *this;
            else if (base_type::t.second.size() == 3 && base_type::t.second[0] == v.u.size() && base_type::t.second[1] == v.u[0].size() && base_type::t.second[2] == v.u[0][0].size()) {
                for (std::size_t i = 0; i < v.u.size(); ++i)
                    for (std::size_t j = 0; j < v.u[i].size(); ++j)
                        for (std::size_t k = 0; k < v.u[i][j].size(); ++k)
                            if (base_type::t.first[(i * v.u[i].size() + j) * v.u[i][j].size() + k] != v.u[i][j][k])
                                return false;
                return true;
            } else
                return false;
        }
};

inline alps::hdf5::oarchive & serialize(alps::hdf5::oarchive & ar, std::string const & p, enum_type const & v) {
    switch (v) {
        case PLUS: ar << alps::make_pvp(p, std::string("plus")); break;
        case MINUS: ar << alps::make_pvp(p, std::string("minus")); break;
    }
    return ar;
}
inline alps::hdf5::iarchive & serialize(alps::hdf5::iarchive & ar, std::string const & p, enum_type & v) {
    std::string s;
    ar >> alps::make_pvp(p, s);
    v = (s == "plus" ? PLUS : MINUS);
    return ar;
}

template<typename T> struct creator {
    typedef T base_type;
    static base_type random() {
        base_type value;
        initialize(value);
        return value;
    }
    static base_type empty() { return base_type(); }
    static base_type special() { return base_type(); }
    static base_type random(alps::hdf5::iarchive & iar) { return base_type(); }
    static base_type empty(alps::hdf5::iarchive & iar) { return base_type(); }
    static base_type special(alps::hdf5::iarchive & iar) { return base_type(); }
};
template<typename T> struct destructor {
    static void apply(T & value) {}
};
template<typename T> bool equal(T const & a, T const & b) {
    return a == b;
}

#define HDF5_DEFINE_VECTOR_TYPE(C)                                                                 \
template<typename T> struct creator< C <T> > {                                                     \
    typedef C <T> base_type;                                                                       \
    static base_type random() {                                                                    \
        base_type value(VECTOR_SIZE);                                                              \
        for (std::size_t i = 0; i < VECTOR_SIZE; ++i)                                              \
            initialize(value[i]);                                                                  \
        return value;                                                                              \
    }                                                                                              \
    static base_type empty() { return base_type(); }                                               \
    static base_type special() { return base_type(); }                                             \
    static base_type random(alps::hdf5::iarchive & iar) { return base_type(); }                    \
    static base_type empty(alps::hdf5::iarchive & iar) { return base_type(); }                     \
    static base_type special(alps::hdf5::iarchive & iar) { return base_type(); }                   \
};                                                                                                 \
template<typename T> bool equal( C <T> const & a,  C <T> const & b) {                              \
    return std::equal(&a[0], &a[0] + a.size(), &b[0]);                                             \
}                                                                                                  \
template<typename T, typename U> struct creator< C < std::pair<T, U> > > {                         \
    typedef C < std::pair<T, U> > base_type;                                                       \
    static base_type random() {                                                                    \
        base_type value(VECTOR_SIZE);                                                              \
        for (std::size_t i = 0; i < VECTOR_SIZE; ++i)                                              \
            value[i] = creator<std::pair<T, U> >::random();                                        \
        return value;                                                                              \
    }                                                                                              \
    static base_type empty() { return base_type(); }                                               \
    static base_type special() { return base_type(); }                                             \
    static base_type random(alps::hdf5::iarchive & iar) { return base_type(); }                    \
    static base_type empty(alps::hdf5::iarchive & iar) { return base_type(); }                     \
    static base_type special(alps::hdf5::iarchive & iar) { return base_type(); }                   \
};                                                                                                 \
template<typename T, typename U> struct destructor<C < std::pair<T, U> > > {                       \
    static void apply(C < std::pair<T, U> > & value) {                                             \
        for (std::size_t i = 0; i < value.size(); ++i)                                             \
            destructor<std::pair<T, U> >::apply(value[i]);                                         \
    }                                                                                              \
};                                                                                                 \
template<typename T, typename U> bool equal(                                                       \
    C < std::pair<T, U> > const & a,  C < std::pair<T, U> > const & b                              \
) {                                                                                                \
    if (a.size() != b.size())                                                                      \
        return false;                                                                              \
    for (std::size_t i = 0; i < a.size(); ++i)                                                     \
        if (!equal(a[i], b[i]))                                                                    \
            return false;                                                                          \
    return true;                                                                                   \
}
HDF5_DEFINE_VECTOR_TYPE(std::vector)
HDF5_DEFINE_VECTOR_TYPE(std::valarray)
HDF5_DEFINE_VECTOR_TYPE(boost::numeric::ublas::vector)
#undef HDF5_DEFINE_VECTOR_TYPE

template<typename T, typename U> struct creator<std::pair<T *, std::vector<U> > > {
    typedef std::pair<T *, std::vector<U> > base_type;
    static base_type random() {
        base_type value = std::make_pair(new typename boost::remove_const<T>::type[VECTOR_SIZE], std::vector<U>(1, VECTOR_SIZE));
        for (std::size_t i = 0; i < VECTOR_SIZE; ++i)
            if (boost::is_scalar<T>::value)
                initialize(const_cast<typename boost::remove_const<T>::type &>(value.first[i]));
            else
                const_cast<typename boost::remove_const<T>::type &>(value.first[i]) = creator<T>::random();
        return value;
    }
    static base_type empty() {
        return std::make_pair(static_cast<typename boost::remove_const<T>::type *>(NULL), std::vector<U>());
    }
    static base_type special() {
        base_type value = std::make_pair(new typename boost::remove_const<T>::type[MATRIX_SIZE * MATRIX_SIZE * MATRIX_SIZE], std::vector<U>(3, MATRIX_SIZE));
        for (std::size_t i = 0; i < MATRIX_SIZE * MATRIX_SIZE * MATRIX_SIZE; ++i)
            if (boost::is_scalar<T>::value)
                initialize(const_cast<typename boost::remove_const<T>::type &>(value.first[i]));
            else
                const_cast<typename boost::remove_const<T>::type &>(value.first[i]) = creator<T>::random();
        return value;
    }
    static base_type random(alps::hdf5::iarchive & iar) {
        return std::make_pair(new typename boost::remove_const<T>::type[VECTOR_SIZE], std::vector<U>(1, VECTOR_SIZE)); 
    }
    static base_type empty(alps::hdf5::iarchive & iar) {
        return std::make_pair(static_cast<typename boost::remove_const<T>::type *>(NULL), std::vector<U>()); 
    }
    static base_type special(alps::hdf5::iarchive & iar) {
        return std::make_pair(new typename boost::remove_const<T>::type[MATRIX_SIZE * MATRIX_SIZE * MATRIX_SIZE], std::vector<U>(3, MATRIX_SIZE));
    }
};
template<typename T, typename U> struct destructor<std::pair<T *, std::vector<U> > > {
    static void apply(std::pair<T *, std::vector<U> > & value) {
        if (value.second.size())
            delete[] value.first;
    }
};
template<typename T, typename U> bool equal(std::pair<T *, std::vector<U> > const & a, std::pair<T *, std::vector<U> > const & b) {
    if (a.second.size() == b.second.size() && std::equal(a.second.begin(), a.second.end(), b.second.begin())) {
        for (std::size_t i = 0; a.second.size() && i < std::accumulate(a.second.begin(), a.second.end(), 1, std::multiplies<hsize_t>()); ++i)
            if (!equal(a.first[i], b.first[i]))
                return false;
        return true;
    } else
        return false;
}

template<typename T, typename A> struct creator<boost::multi_array<T, 2, A> > {
    typedef boost::multi_array<T, 2, A> base_type;
    static base_type random() {
        base_type value(boost::extents[MATRIX_SIZE][MATRIX_SIZE]);
        for (std::size_t i = 0; i < MATRIX_SIZE; ++i)
            for (std::size_t j = 0; j < MATRIX_SIZE; ++j)
                if (boost::is_scalar<T>::value)
                    initialize(value[i][j]);
                else
                    value[i][j] = creator<T>::random();
        return value;
    }
    static base_type empty() { return base_type(boost::extents[2][2]); }
    static base_type special() { return base_type(boost::extents[2][2]); }
    static base_type random(alps::hdf5::iarchive & iar) { return base_type(boost::extents[2][2]); }
    static base_type empty(alps::hdf5::iarchive & iar) { return base_type(boost::extents[2][2]); }
    static base_type special(alps::hdf5::iarchive & iar) { return base_type(boost::extents[2][2]); }
};
template<typename T, typename A> bool equal(boost::multi_array<T, 2, A> const & a,  boost::multi_array<T, 2, A> const & b) {
    for (std::size_t i = 0; i < MATRIX_SIZE; ++i)
        for (std::size_t j = 0; j < MATRIX_SIZE; ++j)
            if (!equal(a[i][j], b[i][j]))
                return false;
    return true;
}

template<typename T> struct creator<boost::numeric::ublas::matrix<T, boost::numeric::ublas::column_major> > {
    typedef boost::numeric::ublas::matrix<T, boost::numeric::ublas::column_major> base_type;
    static base_type random() {
        base_type value (MATRIX_SIZE, MATRIX_SIZE);
        for (std::size_t i = 0; i < MATRIX_SIZE; ++i)
            for (std::size_t j = 0; j < MATRIX_SIZE; ++j)
                if (boost::is_scalar<T>::value)
                    initialize(value(i, j));
                else
                    value(i, j) = creator<T>::random();
        return value;
    }
    static base_type empty() { return base_type(MATRIX_SIZE, MATRIX_SIZE); }
    static base_type special() { return base_type(MATRIX_SIZE, MATRIX_SIZE); }
    static base_type random(alps::hdf5::iarchive & iar) { return base_type(MATRIX_SIZE, MATRIX_SIZE); }
    static base_type empty(alps::hdf5::iarchive & iar) { return base_type(MATRIX_SIZE, MATRIX_SIZE); }
    static base_type special(alps::hdf5::iarchive & iar) { return base_type(MATRIX_SIZE, MATRIX_SIZE); }
};
template<typename T> bool equal(boost::numeric::ublas::matrix<T, boost::numeric::ublas::column_major> const & a, boost::numeric::ublas::matrix<T, boost::numeric::ublas::column_major> const & b) {
    for (std::size_t i = 0; i < MATRIX_SIZE; ++i)
        for (std::size_t j = 0; j < MATRIX_SIZE; ++j)
            if (!equal(a(i, j), b(i, j)))
                return false;
    return true;
}

#define HDF5_DEFINE_VECTOR_VECTOR_TYPE(C, D)                                                       \
template<typename T> struct creator< C < D <T> > > {                                               \
    typedef C < D <T> > base_type;                                                                 \
    static base_type random() {                                                                    \
        base_type value(MATRIX_SIZE);                                                              \
        for (std::size_t i = 0; i < value.size(); ++i) {                                           \
            value[i] = D <T>(MATRIX_SIZE);                                                         \
            for (std::size_t j = 0; j < value[i].size(); ++j)                                      \
                initialize(value[i][j]);                                                           \
        }                                                                                          \
        return value;                                                                              \
    }                                                                                              \
    static base_type empty() { return base_type(); }                                               \
    static base_type special() {                                                                   \
        base_type value(MATRIX_SIZE);                                                              \
        for (std::size_t i = 0; i < value.size(); ++i) {                                           \
            value[i] = D <T>(1 + static_cast<std::size_t>(rng()) % (2 * MATRIX_SIZE));             \
            for (std::size_t j = 0; j < value[i].size(); ++j)                                      \
                initialize(value[i][j]);                                                           \
        }                                                                                          \
        return value;                                                                              \
    }                                                                                              \
    static base_type random(alps::hdf5::iarchive & iar) { return base_type(); }                    \
    static base_type empty(alps::hdf5::iarchive & iar) { return base_type(); }                     \
    static base_type special(alps::hdf5::iarchive & iar) { return base_type(); }                   \
};                                                                                                 \
template<typename T> bool equal( C < D <T> > const & a,  C < D <T> > const & b) {                  \
    for (std::size_t i = 0; i < a.size(); ++i)                                                     \
        if (!std::equal(&a[i][0], &a[i][0] + a[i].size(), &b[i][0]))                               \
            return false;                                                                          \
    return true;                                                                                   \
}
HDF5_DEFINE_VECTOR_VECTOR_TYPE(std::vector, std::vector)
HDF5_DEFINE_VECTOR_VECTOR_TYPE(std::valarray, std::vector)
HDF5_DEFINE_VECTOR_VECTOR_TYPE(std::vector, std::valarray)
HDF5_DEFINE_VECTOR_VECTOR_TYPE(std::valarray, std::valarray)
HDF5_DEFINE_VECTOR_VECTOR_TYPE(std::valarray, boost::numeric::ublas::vector)
HDF5_DEFINE_VECTOR_VECTOR_TYPE(boost::numeric::ublas::vector, std::valarray)
HDF5_DEFINE_VECTOR_VECTOR_TYPE(boost::numeric::ublas::vector, boost::numeric::ublas::vector)
HDF5_DEFINE_VECTOR_VECTOR_TYPE(boost::numeric::ublas::vector, std::vector)
HDF5_DEFINE_VECTOR_VECTOR_TYPE(std::vector, boost::numeric::ublas::vector)
#undef HDF5_DEFINE_VECTOR_VECTOR_TYPE

#define HDF5_DEFINE_VECTOR_VECTOR_VECTOR_TYPE(C, D, E)                                             \
template<typename T> struct creator< C < D < E <T> > > > {                                         \
    typedef C < D < E <T> > > base_type;                                                           \
    static base_type random() {                                                                    \
        base_type value(MATRIX_SIZE);                                                              \
        for (std::size_t i = 0; i < value.size(); ++i) {                                           \
            value[i] = D < E <T> >(MATRIX_SIZE);                                                   \
            for (std::size_t j = 0; j < value[i].size(); ++j) {                                    \
                value[i][j] = D <T>(MATRIX_SIZE);                                                  \
                for (std::size_t k = 0; k < value[i][j].size(); ++k)                               \
                     initialize(value[i][j][k]);                                                   \
            }                                                                                      \
        }                                                                                          \
        return value;                                                                              \
    }                                                                                              \
    static base_type empty() { return base_type(); }                                               \
    static base_type special() {                                                                   \
        base_type value(MATRIX_SIZE);                                                              \
        for (std::size_t i = 0; i < value.size(); ++i) {                                           \
            value[i] = D < E <T> >(1 + static_cast<std::size_t>(rng()) % (2 * MATRIX_SIZE));       \
            for (std::size_t j = 0; j < value[i].size(); ++j) {                                    \
                value[i][j] = E <T>(1 + static_cast<std::size_t>(rng()) % (2 * MATRIX_SIZE));      \
                for (std::size_t k = 0; k < value[i][j].size(); ++k)                               \
                    initialize(value[i][j][k]);                                                    \
            }                                                                                      \
        }                                                                                          \
        return value;                                                                              \
    }                                                                                              \
    static base_type random(alps::hdf5::iarchive & iar) { return base_type(); }                    \
    static base_type empty(alps::hdf5::iarchive & iar) { return base_type(); }                     \
    static base_type special(alps::hdf5::iarchive & iar) { return base_type(); }                   \
};                                                                                                 \
template<typename T> bool equal( C < D < E <T> > > const & a,  C < D < E <T> > > const & b) {      \
    for (std::size_t i = 0; i < a.size(); ++i)                                                     \
        for (std::size_t j = 0; j < a[i].size(); ++j)                                              \
            if (!std::equal(&a[i][j][0], &a[i][j][0] + a[i][j].size(), &b[i][j][0]))               \
                return false;                                                                      \
    return true;                                                                                   \
}
HDF5_DEFINE_VECTOR_VECTOR_VECTOR_TYPE(std::vector, std::vector, std::vector)
#undef HDF5_DEFINE_VECTOR_VECTOR_VECTOR_TYPE

template<typename T> bool equal(T * const & a, T * const & b, std::size_t size) {
    for (std::size_t i = 0; i < size; ++i)
        if (!equal(a[i], b[i]))
            return false;
    return true;
}
