/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2010 - 2012 by Lukas Gamper <gamperl@gmail.com>                   *
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

#include <alps/hdf5/archive.hpp>
#include <alps/hdf5/pointer.hpp>
#include <alps/hdf5/map.hpp>
#include <alps/hdf5/pair.hpp>
#include <alps/hdf5/array.hpp>
#include <alps/hdf5/tuple.hpp>
#include <alps/hdf5/vector.hpp>
#include <alps/hdf5/complex.hpp>
#include <alps/hdf5/valarray.hpp>
#include <alps/hdf5/multi_array.hpp>
#include <alps/hdf5/matrix.hpp>
#include <alps/hdf5/shared_array.hpp>
#include <alps/hdf5/ublas/matrix.hpp>
#include <alps/hdf5/ublas/vector.hpp>

#include <deque>
#include <numeric>
#include <iostream>
#include <algorithm>

#include <boost/random.hpp>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/tuple/tuple_comparison.hpp>

template<class T> class custom_allocator : public std::allocator<T> {
    protected:
        typedef std::allocator<T> alloc_t;

    public:
        typedef typename alloc_t::pointer pointer;
        typedef typename alloc_t::const_pointer const_pointer;
        typedef typename alloc_t::reference reference;
        typedef typename alloc_t::const_reference const_reference;

        typedef typename alloc_t::value_type value_type;
        typedef typename alloc_t::size_type size_type;
        typedef typename alloc_t::difference_type difference_type;

        pointer allocate(size_type n) {
            return alloc_t::allocate(n);
        }

        void deallocate(pointer p, size_type n) {
            alloc_t::deallocate(p, n);
        }

        template <typename T2> struct rebind {
            typedef custom_allocator<T2> other;
        };

        void construct(pointer p, const_reference val) {
            alloc_t::construct(p, val);
        }

        void destroy(pointer p) {
            alloc_t::destroy(p);
        }
};

typedef enum { PLUS, MINUS } enum_type;
typedef enum { PLUS_VEC, MINUS_VEC } enum_vec_type;
template<typename T> struct creator;
template<typename T> class userdefined_class;
template<typename T, typename U> class cast_type;

double rng() {
    static boost::mt19937 rng(SEED);
    static boost::uniform_real<> dist_real(0., 1e9);
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
template<typename T, typename A> void initialize(std::vector<T, A> & v) {
    v = creator<std::vector<T, A> >::random();
}
template<typename A> void initialize(std::vector<bool, A> & v) {
    v = creator<std::vector<bool, A> >::random();
}
template<typename T, std::size_t N> void initialize(boost::array<T, N> & v) {
    for (typename boost::array<T, N>::iterator it = v.begin(); it != v.end(); ++it)
        initialize(*it);
}
template<
    int N, typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9
> void initialize_tuple_value(boost::tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> & v, boost::false_type) {
    using boost::get;
    initialize(get<N>(v));
}
template<
    int N, typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9
> void initialize_tuple_value(boost::tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> & v, boost::true_type) {}
template<
    typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9
> void initialize(boost::tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> & v) {
    initialize_tuple_value<0, T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>(v, typename boost::is_same<T0, boost::tuples::null_type>::type());
    initialize_tuple_value<1, T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>(v, typename boost::is_same<T1, boost::tuples::null_type>::type());
    initialize_tuple_value<2, T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>(v, typename boost::is_same<T2, boost::tuples::null_type>::type());
    initialize_tuple_value<3, T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>(v, typename boost::is_same<T3, boost::tuples::null_type>::type());
    initialize_tuple_value<4, T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>(v, typename boost::is_same<T4, boost::tuples::null_type>::type());
    initialize_tuple_value<5, T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>(v, typename boost::is_same<T5, boost::tuples::null_type>::type());
    initialize_tuple_value<6, T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>(v, typename boost::is_same<T6, boost::tuples::null_type>::type());
    initialize_tuple_value<7, T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>(v, typename boost::is_same<T7, boost::tuples::null_type>::type());
    initialize_tuple_value<8, T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>(v, typename boost::is_same<T8, boost::tuples::null_type>::type());
    initialize_tuple_value<9, T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>(v, typename boost::is_same<T9, boost::tuples::null_type>::type());
}
template<typename T, typename A> void initialize(boost::multi_array<T, 1, A> & v) {
    v.resize(boost::extents[VECTOR_SIZE]);
    v = creator<boost::multi_array<T, 1, A> >::random();
}
template<typename T,  typename A> void initialize(alps::multi_array<T, 1, A> & v) {
    v.resize(boost::extents[VECTOR_SIZE]);
    v = creator<alps::multi_array<T, 1, A> >::random();
}
template<typename T, typename A> void initialize(boost::multi_array<T, 2, A> & v) {
    v.resize(boost::extents[MATRIX_SIZE][MATRIX_SIZE]);
    v = creator<boost::multi_array<T, 2, A> >::random();
}
template<typename T,  typename A> void initialize(alps::multi_array<T, 2, A> & v) {
    v.resize(boost::extents[MATRIX_SIZE][MATRIX_SIZE]);
    v = creator<alps::multi_array<T, 2, A> >::random();
}
template<typename T, typename A> void initialize(boost::multi_array<T, 3, A> & v) {
    v.resize(boost::extents[MATRIX_SIZE][MATRIX_SIZE][MATRIX_SIZE]);
    v = creator<boost::multi_array<T, 3, A> >::random();
}
template<typename T,  typename A> void initialize(alps::multi_array<T, 3, A> & v) {
    v.resize(boost::extents[MATRIX_SIZE][MATRIX_SIZE][MATRIX_SIZE]);
    v = creator<alps::multi_array<T, 3, A> >::random();
}
template<typename T, typename A> void initialize(boost::multi_array<T, 4, A> & v) {
    v.resize(boost::extents[MATRIX_SIZE][MATRIX_SIZE][MATRIX_SIZE][MATRIX_SIZE]);
    v = creator<boost::multi_array<T, 4, A> >::random();
}
template<typename T,  typename A> void initialize(alps::multi_array<T, 4, A> & v) {
    v.resize(boost::extents[MATRIX_SIZE][MATRIX_SIZE][MATRIX_SIZE][MATRIX_SIZE]);
    v = creator<alps::multi_array<T, 4, A> >::random();
}

void initialize(enum_type & v) {
    v = static_cast<std::size_t>(rng()) % 2 == 0 ? PLUS : MINUS;
}
void initialize(enum_vec_type & v) {
    v = static_cast<std::size_t>(rng()) % 2 == 0 ? PLUS_VEC : MINUS_VEC;
}

template<typename T> void initialize(userdefined_class<T> & v);

template<typename T, typename U> void initialize(cast_type<T, U> & v);

template<typename T> bool equal(T const & a, T const & b);

template<typename T> bool equal(T * const & a, T * const & b, std::size_t size);

template<typename T> class userdefined_class {
    public:
        userdefined_class(): b(VECTOR_SIZE) {
            initialize(a);
            for (std::size_t i = 0; i < VECTOR_SIZE; ++i)
                initialize(b[i]);
            initialize(c);
        }
        void save(alps::hdf5::archive & ar) const {
            ar
                << alps::make_pvp("a", a)
                << alps::make_pvp("b", b)
                << alps::make_pvp("c", c)
            ;
        }
        void load(alps::hdf5::archive & ar) { 
            ar
                >> alps::make_pvp("a", a)
                >> alps::make_pvp("b", b)
                >> alps::make_pvp("c", c)
            ;
        }
        bool operator==(userdefined_class<T> const & v) const {
            return a == v.a && b.size() == v.b.size() && (b.size() == 0 || std::equal(b.begin(), b.end(), v.b.begin())) && c == v.c;
        }
    private:
        T a;
        std::vector<T> b;
        enum_type c;
};

template<typename T> void initialize(userdefined_class<T> & v) {
    v = userdefined_class<T>();
}

template<typename T, typename U> class cast_type_base {
    public:
        cast_type_base(T const & v = T()): has_u(false), t(v) {}
        void save(alps::hdf5::archive & ar) const {
            ar
                << alps::make_pvp("t", t)
            ;
        }
        void load(alps::hdf5::archive & ar) { 
            has_u = true;
            ar
                >> alps::make_pvp("t", u)
            ;
        }
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
            return compare(v, typename boost::mpl::and_<
                  typename boost::mpl::or_<typename boost::is_same<T, double>::type, typename boost::is_same<T, float>::type>::type
                , typename boost::mpl::or_<typename boost::is_same<U, double>::type, typename boost::is_same<U, float>::type>::type
            >::type());
        }
    private:
        bool compare(cast_type<T, U> const & v, boost::mpl::true_) const {
            U diff = (base_type::has_u ? base_type::u : alps::cast<U>(base_type::t)) - (v.has_u ? v.u : alps::cast<U>(v.t));
            return (diff > 0 ? diff : -diff) / ((base_type::has_u ? base_type::u : alps::cast<U>(base_type::t)) + (v.has_u ? v.u : alps::cast<U>(v.t))) / 2 < 1e-4;
        }
        bool compare(cast_type<T, U> const & v, boost::mpl::false_) const {
            return (base_type::has_u ? base_type::u : alps::cast<U>(base_type::t)) == (v.has_u ? v.u : alps::cast<U>(v.t));
        }
};

template<typename T, typename U> void initialize(cast_type<T, U> & v) {
    v = cast_type<T, U>();
}

#define HDF5_DEFINE_VECTOR_VECTOR_CAST_TYPE(C, D)                                                          \
template<typename T, typename U> class cast_type< C <T>, D <U> >                                           \
    : public cast_type_base< C <T>, D <U> >                                                                \
{                                                                                                          \
    public:                                                                                                \
        typedef cast_type_base<C <T>, D <U> > base_type;                                                   \
        cast_type(): base_type(creator< C <T> >::random()) {}                                              \
        bool operator==(cast_type< C <T>, D <U> > const & vc)  const {                                     \
             cast_type< C <T>, D <U> >& v = const_cast<cast_type< C <T>, D <U> > &>(vc);                   \
             base_type& nonconstbase(const_cast<cast_type< C <T>, D <U> > &>(*this));                      \
            if (base_type::has_u && !v.has_u)                                                              \
                return base_type::u.size() == v.t.size() && (                                              \
                       v.t.size() == 0                                                                     \
                    || std::equal(&nonconstbase.u[0], &nonconstbase.u[0] +nonconstbase.u.size(), &v.t[0])  \
                );                                                                                         \
            else if (!base_type::has_u && v.has_u)                                                         \
                return base_type::t.size() == v.u.size() && (                                              \
                       v.u.size() == 0                                                                     \
                    || std::equal(&nonconstbase.t[0], &nonconstbase.t[0] + nonconstbase.t.size(), &v.u[0]) \
                );                                                                                         \
            else                                                                                           \
                return false;                                                                              \
        }                                                                                                  \
};
HDF5_DEFINE_VECTOR_VECTOR_CAST_TYPE(std::valarray, std::vector)
HDF5_DEFINE_VECTOR_VECTOR_CAST_TYPE(std::vector, std::valarray)
HDF5_DEFINE_VECTOR_VECTOR_CAST_TYPE(std::deque, std::vector)
HDF5_DEFINE_VECTOR_VECTOR_CAST_TYPE(std::vector, std::deque)
HDF5_DEFINE_VECTOR_VECTOR_CAST_TYPE(std::deque, std::valarray)
HDF5_DEFINE_VECTOR_VECTOR_CAST_TYPE(std::valarray, std::deque)
HDF5_DEFINE_VECTOR_VECTOR_CAST_TYPE(std::valarray, boost::numeric::ublas::vector)
HDF5_DEFINE_VECTOR_VECTOR_CAST_TYPE(boost::numeric::ublas::vector, std::valarray)
HDF5_DEFINE_VECTOR_VECTOR_CAST_TYPE(boost::numeric::ublas::vector, std::vector)
HDF5_DEFINE_VECTOR_VECTOR_CAST_TYPE(std::vector, boost::numeric::ublas::vector)
HDF5_DEFINE_VECTOR_VECTOR_CAST_TYPE(std::deque, boost::numeric::ublas::vector)
HDF5_DEFINE_VECTOR_VECTOR_CAST_TYPE(boost::numeric::ublas::vector, std::deque)

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

namespace alps {
    namespace hdf5 {

        void save(
              alps::hdf5::archive & ar
            , std::string const & path
            , enum_type const & value
            , std::vector<std::size_t> size = std::vector<std::size_t>()
            , std::vector<std::size_t> chunk = std::vector<std::size_t>()
            , std::vector<std::size_t> offset = std::vector<std::size_t>()
        ) {
            switch (value) {
                case PLUS: ar << alps::make_pvp(path, std::string("plus")); break;
                case MINUS: ar << alps::make_pvp(path, std::string("minus")); break;
            }
        }
        void load(
              alps::hdf5::archive & ar
            , std::string const & path
            , enum_type & value
            , std::vector<std::size_t> chunk = std::vector<std::size_t>()
            , std::vector<std::size_t> offset = std::vector<std::size_t>()
        ) {
            std::string s;
            ar >> alps::make_pvp(path, s);
            value = (s == "plus" ? PLUS : MINUS);
        }

        template<> struct scalar_type<enum_vec_type> {
            typedef boost::int_t<sizeof(enum_vec_type) * 8>::exact type;
        };

        template<> struct is_continuous<enum_vec_type> : public boost::true_type {};

        template<> struct has_complex_elements<enum_vec_type> : public boost::false_type {};

        namespace detail {

            template<> struct get_extent<enum_vec_type> {
                static std::vector<std::size_t> apply(enum_vec_type const & value) {
                    return std::vector<std::size_t>();
                }
            };

            template<> struct set_extent<enum_vec_type> {
                static void apply(enum_vec_type &, std::vector<std::size_t> const &) {}
            };

            template<> struct is_vectorizable<enum_vec_type> {
                static bool apply(enum_vec_type const & value) {
                    return true;
                }
            };

            template<> struct get_pointer<enum_vec_type> {
                static scalar_type<enum_vec_type>::type * apply(enum_vec_type & value) {
                    return reinterpret_cast<scalar_type<enum_vec_type>::type *>(&value);
                }
            };
        
            template<> struct get_pointer<enum_vec_type const> {
                static scalar_type<enum_vec_type>::type const * apply(enum_vec_type const & value) {
                    return reinterpret_cast<scalar_type<enum_vec_type>::type const *>(&value);
                }
            };
        }

        void save(
              alps::hdf5::archive & ar
            , std::string const & path
            , enum_vec_type const & value
            , std::vector<std::size_t> size = std::vector<std::size_t>()
            , std::vector<std::size_t> chunk = std::vector<std::size_t>()
            , std::vector<std::size_t> offset = std::vector<std::size_t>()
        ) {
            if (size.size() == 0) {
                size.push_back(1);
                chunk.push_back(1);
                offset.push_back(0);
            }
            ar.write(path, (scalar_type<enum_vec_type>::type const *)get_pointer(value), size, chunk, offset);
        }
        
        void load(
              alps::hdf5::archive & ar
            , std::string const & path
            , enum_vec_type & value
            , std::vector<std::size_t> chunk = std::vector<std::size_t>()
            , std::vector<std::size_t> offset = std::vector<std::size_t>()
        ) {
            if (chunk.size() == 0) {
                chunk.push_back(1);
                offset.push_back(0);
            }
            ar.read(path, (scalar_type<enum_vec_type>::type *)get_pointer(value), chunk, offset);
        }

    }
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
    template<typename X> static base_type random(X const &) { return base_type(); }
    template<typename X> static base_type empty(X const &) { return base_type(); }
    template<typename X> static base_type special(X const &) { return base_type(); }
};
template<typename T> struct destructor {
    static void apply(T & value) {}
};
template<typename T> bool equal(T const & a, T const & b) {
    return a == b;
}


#define HDF5_DEFINE_MULTI_ARRAY_TYPE(P, C)                                                         \
template<typename T, typename A> struct creator< C < P ::multi_array<T, 1, A> > > {                \
    typedef C < P ::multi_array<T, 1, A> > base_type;                                              \
    static base_type random() {                                                                    \
        base_type value(                                                                           \
            VECTOR_SIZE, P ::multi_array<T, 1, A>(boost::extents[MATRIX_SIZE])                     \
        );                                                                                         \
        for (std::size_t i = 0; i < VECTOR_SIZE; ++i)                                              \
            for (std::size_t j = 0; j < MATRIX_SIZE; ++j)                                          \
                if (boost::is_scalar<T>::value)                                                    \
                    initialize(value[i][j]);                                                       \
                else                                                                               \
                    value[i][j] = creator<T>::random();                                            \
        return value;                                                                              \
    }                                                                                              \
    static base_type empty() { return base_type(); }                                               \
    static base_type special() { return base_type(); }                                             \
    template<typename X> static base_type random(X const &) { return base_type(); }                \
    template<typename X> static base_type empty(X const &) { return base_type(); }                 \
    template<typename X> static base_type special(X const &) { return base_type(); }               \
};                                                                                                 \
template<typename T, typename A> struct creator< C < P ::multi_array<T, 2, A> > > {                \
    typedef C < P ::multi_array<T, 2, A> > base_type;                                              \
    static base_type random() {                                                                    \
        base_type value(                                                                           \
            VECTOR_SIZE, P ::multi_array<T, 2, A>(boost::extents[MATRIX_SIZE][MATRIX_SIZE])        \
        );                                                                                         \
        for (std::size_t i = 0; i < VECTOR_SIZE; ++i)                                              \
            for (std::size_t j = 0; j < MATRIX_SIZE; ++j)                                          \
                for (std::size_t k = 0; k < MATRIX_SIZE; ++k)                                      \
                    if (boost::is_scalar<T>::value)                                                \
                        initialize(value[i][j][k]);                                                \
                    else                                                                           \
                        value[i][j][k] = creator<T>::random();                                     \
        return value;                                                                              \
    }                                                                                              \
    static base_type empty() { return base_type(); }                                               \
    static base_type special() { return base_type(); }                                             \
    template<typename X> static base_type random(X const &) { return base_type(); }                \
    template<typename X> static base_type empty(X const &) { return base_type(); }                 \
    template<typename X> static base_type special(X const &) { return base_type(); }               \
};                                                                                                 \
template<typename T, typename A> struct creator< C < P ::multi_array<T, 3, A> > > {                \
    typedef C < P ::multi_array<T, 3, A> > base_type;                                              \
    static base_type random() {                                                                    \
        base_type value(                                                                           \
            VECTOR_SIZE, P ::multi_array<T, 3, A>(                                                 \
                boost::extents[MATRIX_SIZE][MATRIX_SIZE][MATRIX_SIZE]                              \
            )                                                                                      \
        );                                                                                         \
        for (std::size_t i = 0; i < VECTOR_SIZE; ++i)                                              \
            for (std::size_t j = 0; j < MATRIX_SIZE; ++j)                                          \
                for (std::size_t k = 0; k < MATRIX_SIZE; ++k)                                      \
                    for (std::size_t l = 0; l < MATRIX_SIZE; ++l)                                  \
                        if (boost::is_scalar<T>::value)                                            \
                            initialize(value[i][j][k][l]);                                         \
                        else                                                                       \
                            value[i][j][k][l] = creator<T>::random();                              \
        return value;                                                                              \
    }                                                                                              \
    static base_type empty() { return base_type(); }                                               \
    static base_type special() { return base_type(); }                                             \
    template<typename X> static base_type random(X const &) { return base_type(); }                \
    template<typename X> static base_type empty(X const &) { return base_type(); }                 \
    template<typename X> static base_type special(X const &) { return base_type(); }               \
};                                                                                                 \
template<typename T, typename A> struct creator< C < P ::multi_array<T, 4, A> > > {                \
    typedef C < P ::multi_array<T, 4, A> > base_type;                                              \
    static base_type random() {                                                                    \
        base_type value(                                                                           \
            VECTOR_SIZE, P ::multi_array<T, 4, A>(                                                 \
                boost::extents[MATRIX_SIZE][MATRIX_SIZE][MATRIX_SIZE][MATRIX_SIZE]                 \
            )                                                                                      \
        );                                                                                         \
        for (std::size_t i = 0; i < VECTOR_SIZE; ++i)                                              \
            for (std::size_t j = 0; j < MATRIX_SIZE; ++j)                                          \
                for (std::size_t k = 0; k < MATRIX_SIZE; ++k)                                      \
                    for (std::size_t l = 0; l < MATRIX_SIZE; ++l)                                  \
                        for (std::size_t m = 0; m < MATRIX_SIZE; ++m)                              \
                            if (boost::is_scalar<T>::value)                                        \
                                initialize(value[i][j][k][l][m]);                                  \
                            else                                                                   \
                                value[i][j][k][l][m] = creator<T>::random();                       \
        return value;                                                                              \
    }                                                                                              \
    static base_type empty() { return base_type(); }                                               \
    static base_type special() { return base_type(); }                                             \
    template<typename X> static base_type random(X const &) { return base_type(); }                \
    template<typename X> static base_type empty(X const &) { return base_type(); }                 \
    template<typename X> static base_type special(X const &) { return base_type(); }               \
};

#undef HDF5_DEFINE_VECTOR_TYPE
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
    template<typename X> static base_type random(X const &) { return base_type(); }                \
    template<typename X> static base_type empty(X const &) { return base_type(); }                 \
    template<typename X> static base_type special(X const &) { return base_type(); }               \
};                                                                                                 \
template<typename T> bool equal( C <T> const & a,  C <T> const & b) {                              \
    return a.size() == b.size() && (a.size() == 0 ||                                               \
        std::equal(&const_cast<C<T>&>(a)[0], &const_cast<C<T>&>(a)[0] + a.size(),                  \
                      &const_cast<C<T>&>(b)[0]));                                                  \
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
    template<typename X> static base_type random(X const &) { return base_type(); }                \
    template<typename X> static base_type empty(X const &) { return base_type(); }                 \
    template<typename X> static base_type special(X const &) { return base_type(); }               \
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
}                                                                                                  \
template<typename T> struct creator< C < alps::numeric::matrix<T> > > {                            \
    typedef C < alps::numeric::matrix<T> > base_type;                                              \
    static base_type random() {                                                                    \
        base_type value(                                                                           \
            VECTOR_SIZE, alps::numeric::matrix<T>(MATRIX_SIZE, MATRIX_SIZE)                        \
        );                                                                                         \
        for (std::size_t i = 0; i < VECTOR_SIZE; ++i)                                              \
            for (std::size_t j = 0; j < MATRIX_SIZE; ++j)                                          \
                for (std::size_t k = 0; k < MATRIX_SIZE; ++k)                                      \
                    if (boost::is_scalar<T>::value)                                                \
                        initialize(value[i](j, k));                                                \
                    else                                                                           \
                        value[i](j, k) = creator<T>::random();                                     \
        return value;                                                                              \
    }                                                                                              \
    static base_type empty() { return base_type(); }                                               \
    static base_type special() { return base_type(); }                                             \
    template<typename X> static base_type random(X const &) { return base_type(); }                \
    template<typename X> static base_type empty(X const &) { return base_type(); }                 \
    template<typename X> static base_type special(X const &) { return base_type(); }               \
};                                                                                                 \
HDF5_DEFINE_MULTI_ARRAY_TYPE(boost, C)                                                             \
HDF5_DEFINE_MULTI_ARRAY_TYPE(alps, C)

HDF5_DEFINE_VECTOR_TYPE(std::vector)
HDF5_DEFINE_VECTOR_TYPE(std::valarray)
HDF5_DEFINE_VECTOR_TYPE(std::deque)
HDF5_DEFINE_VECTOR_TYPE(boost::numeric::ublas::vector)
#undef HDF5_DEFINE_VECTOR_TYPE
#undef HDF5_DEFINE_MULTI_ARRAY_TYPE

template<> struct creator<std::vector<bool> > {
    typedef std::vector<bool> base_type;
    static base_type random() {
        base_type value(VECTOR_SIZE);
        for (std::size_t i = 0; i < VECTOR_SIZE; ++i) {
            bool tmp;
            initialize(tmp);
            value[i] = tmp;
        }
        return value;
    }
    static base_type empty() { return base_type(); }
    static base_type special() { return base_type(); }
    template<typename X> static base_type random(X const &) { return base_type(); }
    template<typename X> static base_type empty(X const &) { return base_type(); }
    template<typename X> static base_type special(X const &) { return base_type(); }
};
template<> bool equal(std::vector<bool> const & a, std::vector<bool> const & b) {
    return a.size() == b.size() && (a.size() == 0 || std::equal(a.begin(), a.end(), b.begin()));
}

template<typename T, typename U> struct creator<std::pair<T, U> > {
    typedef std::pair<T, U> base_type;
    static base_type random() {
        return std::make_pair(creator<T>::random(), creator<U>::random());
    }
    static base_type empty() {
        return std::make_pair(creator<T>::empty(), creator<U>::empty());
    }
    static base_type special() {
        return std::make_pair(creator<T>::special(), creator<U>::special());
    }
    template<typename X> static base_type random(X const & x) {
        return std::make_pair(creator<T>::random(x), creator<U>::random(x));
    }
    template<typename X> static base_type empty(X const & x) {
        return std::make_pair(creator<T>::empty(x), creator<U>::empty(x));
    }
    template<typename X> static base_type special(X const & x) {
        return std::make_pair(creator<T>::special(x), creator<U>::special(x));
    }
};
template<typename T, typename U> bool equal(std::pair<T, U> const & a, std::pair<T, U> const & b) {
    return equal(a.first, b.first) && equal(a.second, b.second);
}

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
    template<typename X> static base_type random(X const &) {
        return std::make_pair(new typename boost::remove_const<T>::type[VECTOR_SIZE], std::vector<U>(1, VECTOR_SIZE)); 
    }
    template<typename X> static base_type empty(X const &) {
        return std::make_pair(static_cast<typename boost::remove_const<T>::type *>(NULL), std::vector<U>()); 
    }
    template<typename X> static base_type special(X const &) {
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
        for (std::size_t i = 0; a.second.size() && i < std::accumulate(a.second.begin(), a.second.end(), std::size_t(1), std::multiplies<std::size_t>()); ++i)
            if (!equal(a.first[i], b.first[i]))
                return false;
        return true;
    } else
        return false;
}

#define HDF5_DEFINE_MULTI_ARRAY_TYPE(P)                                                                                                             \
    template<typename T, typename A> struct creator< P ::multi_array<T, 1, A> > {                                                                   \
        typedef  P ::multi_array<T, 1, A> base_type;                                                                                                \
        static base_type random() {                                                                                                                 \
            base_type value(boost::extents[VECTOR_SIZE]);                                                                                           \
            for (std::size_t i = 0; i < VECTOR_SIZE; ++i)                                                                                           \
                if (boost::is_scalar<T>::value)                                                                                                     \
                    initialize(value[i]);                                                                                                           \
                else                                                                                                                                \
                    value[i] = creator<T>::random();                                                                                                \
            return value;                                                                                                                           \
        }                                                                                                                                           \
        static base_type empty() { return base_type(boost::extents[0]); }                                                                           \
        static base_type special() { return base_type(boost::extents[VECTOR_SIZE]); }                                                               \
        template<typename X> static base_type random(X const &) { return base_type(boost::extents[VECTOR_SIZE]); }                                  \
        template<typename X> static base_type empty(X const &) { return base_type(boost::extents[0]); }                                             \
        template<typename X> static base_type special(X const &) { return base_type(boost::extents[VECTOR_SIZE]); }                                 \
    };                                                                                                                                              \
    template<typename T, typename A> bool equal( P ::multi_array<T, 1, A> const & a,   P ::multi_array<T, 1, A> const & b) {                        \
        if (!std::equal(a.shape(), a.shape() +  P ::multi_array<T, 1, A>::dimensionality, b.shape()))                                               \
            return false;                                                                                                                           \
        for (std::size_t i = 0; i < a.shape()[0]; ++i)                                                                                              \
            if (!equal(a[i], b[i]))                                                                                                                 \
                return false;                                                                                                                       \
        return true;                                                                                                                                \
    }                                                                                                                                               \
                                                                                                                                                    \
    template<typename T, typename A> struct creator< P ::multi_array<T, 2, A> > {                                                                   \
        typedef  P ::multi_array<T, 2, A> base_type;                                                                                                \
        static base_type random() {                                                                                                                 \
            base_type value(boost::extents[MATRIX_SIZE][MATRIX_SIZE]);                                                                              \
            for (std::size_t i = 0; i < MATRIX_SIZE; ++i)                                                                                           \
                for (std::size_t j = 0; j < MATRIX_SIZE; ++j)                                                                                       \
                    if (boost::is_scalar<T>::value)                                                                                                 \
                        initialize(value[i][j]);                                                                                                    \
                    else                                                                                                                            \
                        value[i][j] = creator<T>::random();                                                                                         \
            return value;                                                                                                                           \
        }                                                                                                                                           \
        static base_type empty() { return base_type(boost::extents[0][0]); }                                                                        \
        static base_type special() { return base_type(boost::extents[MATRIX_SIZE][MATRIX_SIZE]); }                                                  \
        template<typename X> static base_type random(X const &) { return base_type(boost::extents[5][5]); }                                         \
        template<typename X> static base_type empty(X const &) { return base_type(boost::extents[0][0]); }                                          \
        template<typename X> static base_type special(X const &) { return base_type(boost::extents[MATRIX_SIZE][MATRIX_SIZE]); }                    \
    };                                                                                                                                              \
    template<typename T, typename A> bool equal( P ::multi_array<T, 2, A> const & a,   P ::multi_array<T, 2, A> const & b) {                        \
        if (!std::equal(a.shape(), a.shape() +  P ::multi_array<T, 2, A>::dimensionality, b.shape()))                                               \
            return false;                                                                                                                           \
        for (std::size_t i = 0; i < a.shape()[0]; ++i)                                                                                              \
            for (std::size_t j = 0; j < a.shape()[1]; ++j)                                                                                          \
                if (!equal(a[i][j], b[i][j]))                                                                                                       \
                    return false;                                                                                                                   \
        return true;                                                                                                                                \
    }                                                                                                                                               \
                                                                                                                                                    \
    template<typename T, typename A> struct creator< P ::multi_array<T, 3, A> > {                                                                   \
        typedef  P ::multi_array<T, 3, A> base_type;                                                                                                \
        static base_type random() {                                                                                                                 \
            base_type value(boost::extents[MATRIX_SIZE][MATRIX_SIZE][MATRIX_SIZE]);                                                                 \
            for (std::size_t i = 0; i < MATRIX_SIZE; ++i)                                                                                           \
                for (std::size_t j = 0; j < MATRIX_SIZE; ++j)                                                                                       \
                    for (std::size_t k = 0; k < MATRIX_SIZE; ++k)                                                                                   \
                        if (boost::is_scalar<T>::value)                                                                                             \
                            initialize(value[i][j][k]);                                                                                             \
                        else                                                                                                                        \
                            value[i][j][k] = creator<T>::random();                                                                                  \
            return value;                                                                                                                           \
        }                                                                                                                                           \
        static base_type empty() { return base_type(boost::extents[0][0][0]); }                                                                     \
        static base_type special() { return base_type(boost::extents[MATRIX_SIZE][MATRIX_SIZE][MATRIX_SIZE]); }                                     \
        template<typename X> static base_type random(X const &) { return base_type(boost::extents[MATRIX_SIZE][MATRIX_SIZE][MATRIX_SIZE]); }        \
        template<typename X> static base_type empty(X const &) { return base_type(boost::extents[0][0][0]); }                                       \
        template<typename X> static base_type special(X const &) { return base_type(boost::extents[MATRIX_SIZE][MATRIX_SIZE][MATRIX_SIZE]); }       \
    };                                                                                                                                              \
    template<typename T, typename A> bool equal( P ::multi_array<T, 3, A> const & a,   P ::multi_array<T, 3, A> const & b) {                        \
        if (!std::equal(a.shape(), a.shape() +  P ::multi_array<T, 3, A>::dimensionality, b.shape()))                                               \
            return false;                                                                                                                           \
        for (std::size_t i = 0; i < a.shape()[0]; ++i)                                                                                              \
            for (std::size_t j = 0; j < a.shape()[1]; ++j)                                                                                          \
                for (std::size_t k = 0; k < a.shape()[2]; ++k)                                                                                      \
                    if (!equal(a[i][j][k], b[i][j][k]))                                                                                             \
                        return false;                                                                                                               \
        return true;                                                                                                                                \
    }                                                                                                                                               \
                                                                                                                                                    \
    template<typename T, typename A> struct creator< P ::multi_array<T, 4, A> > {                                                                   \
        typedef  P ::multi_array<T, 4, A> base_type;                                                                                                \
        static base_type random() {                                                                                                                 \
            base_type value(boost::extents[MATRIX_SIZE][MATRIX_SIZE][MATRIX_SIZE][MATRIX_SIZE]);                                                    \
            for (std::size_t i = 0; i < MATRIX_SIZE; ++i)                                                                                           \
                for (std::size_t j = 0; j < MATRIX_SIZE; ++j)                                                                                       \
                    for (std::size_t k = 0; k < MATRIX_SIZE; ++k)                                                                                   \
                        for (std::size_t l = 0; l < MATRIX_SIZE; ++l)                                                                               \
                            if (boost::is_scalar<T>::value)                                                                                         \
                                initialize(value[i][j][k][l]);                                                                                      \
                            else                                                                                                                    \
                                value[i][j][k][l] = creator<T>::random();                                                                           \
            return value;                                                                                                                           \
        }                                                                                                                                           \
        static base_type empty() { return base_type(boost::extents[0][0][0]); }                                                                     \
        static base_type special() { return base_type(boost::extents[MATRIX_SIZE][MATRIX_SIZE][MATRIX_SIZE]); }                                     \
        template<typename X> static base_type random(X const &) { return base_type(boost::extents[MATRIX_SIZE][MATRIX_SIZE][MATRIX_SIZE]); }        \
        template<typename X> static base_type empty(X const &) { return base_type(boost::extents[0][0][0]); }                                       \
        template<typename X> static base_type special(X const &) { return base_type(boost::extents[MATRIX_SIZE][MATRIX_SIZE][MATRIX_SIZE]); }       \
    };                                                                                                                                              \
    template<typename T, typename A> bool equal( P ::multi_array<T, 4, A> const & a,   P ::multi_array<T, 4, A> const & b) {                        \
        if (!std::equal(a.shape(), a.shape() +  P ::multi_array<T, 4, A>::dimensionality, b.shape()))                                               \
            return false;                                                                                                                           \
        for (std::size_t i = 0; i < a.shape()[0]; ++i)                                                                                              \
            for (std::size_t j = 0; j < a.shape()[1]; ++j)                                                                                          \
                for (std::size_t k = 0; k < a.shape()[2]; ++k)                                                                                      \
                    for (std::size_t l = 0; l < a.shape()[2]; ++l)                                                                                  \
                        if (!equal(a[i][j][k][l], b[i][j][k][l]))                                                                                   \
                            return false;                                                                                                           \
        return true;                                                                                                                                \
    }
HDF5_DEFINE_MULTI_ARRAY_TYPE(boost)
HDF5_DEFINE_MULTI_ARRAY_TYPE(alps)

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
    template<typename X> static base_type random(X const &) { return base_type(MATRIX_SIZE, MATRIX_SIZE); }
    template<typename X> static base_type empty(X const &) { return base_type(MATRIX_SIZE, MATRIX_SIZE); }
    template<typename X> static base_type special(X const &) { return base_type(MATRIX_SIZE, MATRIX_SIZE); }
};
template<typename T> bool equal(boost::numeric::ublas::matrix<T, boost::numeric::ublas::column_major> const & a, boost::numeric::ublas::matrix<T, boost::numeric::ublas::column_major> const & b) {
    for (std::size_t i = 0; i < MATRIX_SIZE; ++i)
        for (std::size_t j = 0; j < MATRIX_SIZE; ++j)
            if (!equal(a(i, j), b(i, j)))
                return false;
    return true;
}

template<typename T> struct creator<alps::numeric::matrix<T> > {
    typedef alps::numeric::matrix<T> base_type;
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
    static base_type empty() { return base_type(); }
    static base_type special() { return base_type(); }
    template<typename X> static base_type random(X const &) { return base_type(MATRIX_SIZE, MATRIX_SIZE); }
    template<typename X> static base_type empty(X const &) { return base_type(); }
    template<typename X> static base_type special(X const &) { return base_type(); }
};
template<typename T> bool equal(alps::numeric::matrix<T> const & a, alps::numeric::matrix<T> const & b) {
    return a == b;
}

#define HDF5_DEFINE_VECTOR_VECTOR_TYPE(C, D)                                                       \
template<typename T> struct creator< C < D <T> > > {                                               \
    typedef C < D <T> > base_type;                                                                 \
    static base_type random() {                                                                    \
        base_type value(MATRIX_SIZE);                                                              \
        for (std::size_t i = 0; i < value.size(); ++i) {                                           \
            value[i].resize(MATRIX_SIZE);                                                          \
            for (std::size_t j = 0; j < value[i].size(); ++j)                                      \
                initialize(value[i][j]);                                                           \
        }                                                                                          \
        return value;                                                                              \
    }                                                                                              \
    static base_type empty() { return base_type(); }                                               \
    static base_type special() {                                                                   \
        base_type value(MATRIX_SIZE);                                                              \
        for (std::size_t i = 0; i < value.size(); ++i) {                                           \
            value[i].resize(1 + static_cast<std::size_t>(rng()) % (2 * MATRIX_SIZE));              \
            for (std::size_t j = 0; j < value[i].size(); ++j)                                      \
                initialize(value[i][j]);                                                           \
        }                                                                                          \
        return value;                                                                              \
    }                                                                                              \
    template<typename X> static base_type random(X const &) { return base_type(); }                \
    template<typename X> static base_type empty(X const &) { return base_type(); }                 \
    template<typename X> static base_type special(X const &) { return base_type(); }               \
};                                                                                                 \
template<typename T> bool equal( C < D <T> > const & a,  C < D <T> > const & b) {                  \
    for (std::size_t i = 0; i < a.size(); ++i)                                                     \
        if (a[i].size() != b[i].size() || (                                                        \
            a[i].size() > 0 &&  !equal(&const_cast<C<D<T> >&>(a)[i][0],                            \
                        &const_cast<C<D<T> >&>(b)[i][0],                                           \
                        a[i].size())                                                               \
        ))                                                                                         \
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
HDF5_DEFINE_VECTOR_VECTOR_TYPE(std::deque, std::deque)
HDF5_DEFINE_VECTOR_VECTOR_TYPE(std::deque, std::vector)
HDF5_DEFINE_VECTOR_VECTOR_TYPE(std::vector, std::deque)
HDF5_DEFINE_VECTOR_VECTOR_TYPE(std::valarray, std::deque)
HDF5_DEFINE_VECTOR_VECTOR_TYPE(std::deque, std::valarray)
HDF5_DEFINE_VECTOR_VECTOR_TYPE(boost::numeric::ublas::vector, std::deque)
HDF5_DEFINE_VECTOR_VECTOR_TYPE(std::deque, boost::numeric::ublas::vector)
#undef HDF5_DEFINE_VECTOR_VECTOR_TYPE

#define HDF5_DEFINE_VECTOR_VECTOR_VECTOR_TYPE(C, D, E)                                                    \
template<typename T> struct creator< C < D < E <T> > > > {                                                \
    typedef C < D < E <T> > > base_type;                                                                  \
    static base_type random() {                                                                           \
        base_type value(MATRIX_SIZE);                                                                     \
        for (std::size_t i = 0; i < value.size(); ++i) {                                                  \
            value[i] = D < E <T> >(MATRIX_SIZE);                                                          \
            for (std::size_t j = 0; j < value[i].size(); ++j) {                                           \
                value[i][j] = D <T>(MATRIX_SIZE);                                                         \
                for (std::size_t k = 0; k < value[i][j].size(); ++k)                                      \
                     initialize(value[i][j][k]);                                                          \
            }                                                                                             \
        }                                                                                                 \
        return value;                                                                                     \
    }                                                                                                     \
    static base_type empty() { return base_type(); }                                                      \
    static base_type special() {                                                                          \
        base_type value(MATRIX_SIZE);                                                                     \
        for (std::size_t i = 0; i < value.size(); ++i) {                                                  \
            value[i] = D < E <T> >(1 + static_cast<std::size_t>(rng()) % (2 * MATRIX_SIZE));              \
            for (std::size_t j = 0; j < value[i].size(); ++j) {                                           \
                value[i][j] = E <T>(1 + static_cast<std::size_t>(rng()) % (2 * MATRIX_SIZE));             \
                for (std::size_t k = 0; k < value[i][j].size(); ++k)                                      \
                    initialize(value[i][j][k]);                                                           \
            }                                                                                             \
        }                                                                                                 \
        return value;                                                                                     \
    }                                                                                                     \
    template<typename X> static base_type random(X const &) { return base_type(); }                       \
    template<typename X> static base_type empty(X const &) { return base_type(); }                        \
    template<typename X> static base_type special(X const &) { return base_type(); }                      \
};                                                                                                        \
template<typename T> bool equal( C < D < E <T> > > const & a,  C < D < E <T> > > const & b) {             \
    for (std::size_t i = 0; i < a.size(); ++i)                                                            \
        for (std::size_t j = 0; j < a[i].size(); ++j)                                                     \
            if (a[i][j].size() != b[i][j].size() || (                                                     \
                a[i][j].size() > 0 && !std::equal(&a[i][j][0], &a[i][j][0] + a[i][j].size(), &b[i][j][0]) \
            ))                                                                                            \
                return false;                                                                             \
    return true;                                                                                          \
}
HDF5_DEFINE_VECTOR_VECTOR_VECTOR_TYPE(std::vector, std::vector, std::vector)
#undef HDF5_DEFINE_VECTOR_VECTOR_VECTOR_TYPE

template<typename T> bool equal(T * const & a, T * const & b, std::size_t size) {
    for (std::size_t i = 0; i < size; ++i)
        if (!equal(a[i], b[i]))
            return false;
    return true;
}
