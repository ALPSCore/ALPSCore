/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/hdf5/archive.hpp>
#include <alps/hdf5/pointer.hpp>
#include <alps/hdf5/map.hpp>
#include <alps/hdf5/pair.hpp>
#include <alps/hdf5/array.hpp>
#include <alps/hdf5/tuple.hpp>
#include <alps/hdf5/vector.hpp>
#include <alps/hdf5/complex.hpp>
#include <alps/hdf5/valarray.hpp>
#include <alps/hdf5/shared_array.hpp>

#include <alps/testing/unique_file.hpp>

#include <deque>
#include <numeric>
#include <iostream>
#include <algorithm>
#include <type_traits>

#include <boost/random.hpp>
#include <boost/multi_array.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/tuple/tuple_comparison.hpp>

#define VECTOR_SIZE 25
#define MATRIX_SIZE 5
#define SEED 42

#ifndef SZIP_COMPRESS
    #define SZIP_COMPRESS false
#endif

#ifndef IS_ATTRIBUTE
    #define IS_ATTRIBUTE false
#endif

#if defined(__FCC_VERSION) && defined(main) // workaround for FCC with SSL2
    extern "C"{
        extern void setrcd_(int *);
    };
#endif

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
> void initialize_tuple_value(boost::tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> & v, std::false_type) {
    using boost::get;
    initialize(get<N>(v));
}
template<
    int N, typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9
> void initialize_tuple_value(boost::tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> & v, std::true_type) {}
template<
    typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9
> void initialize(boost::tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> & v) {
    initialize_tuple_value<0, T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>(v, typename std::is_same<T0, boost::tuples::null_type>::type());
    initialize_tuple_value<1, T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>(v, typename std::is_same<T1, boost::tuples::null_type>::type());
    initialize_tuple_value<2, T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>(v, typename std::is_same<T2, boost::tuples::null_type>::type());
    initialize_tuple_value<3, T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>(v, typename std::is_same<T3, boost::tuples::null_type>::type());
    initialize_tuple_value<4, T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>(v, typename std::is_same<T4, boost::tuples::null_type>::type());
    initialize_tuple_value<5, T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>(v, typename std::is_same<T5, boost::tuples::null_type>::type());
    initialize_tuple_value<6, T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>(v, typename std::is_same<T6, boost::tuples::null_type>::type());
    initialize_tuple_value<7, T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>(v, typename std::is_same<T7, boost::tuples::null_type>::type());
    initialize_tuple_value<8, T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>(v, typename std::is_same<T8, boost::tuples::null_type>::type());
    initialize_tuple_value<9, T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>(v, typename std::is_same<T9, boost::tuples::null_type>::type());
}
template<typename T, typename A> void initialize(boost::multi_array<T, 1, A> & v) {
    v.resize(boost::extents[VECTOR_SIZE]);
    v = creator<boost::multi_array<T, 1, A> >::random();
}
// template<typename T,  typename A> void initialize(alps::multi_array<T, 1, A> & v) {
//     v.resize(boost::extents[VECTOR_SIZE]);
//     v = creator<alps::multi_array<T, 1, A> >::random();
// }
// template<typename T, typename A> void initialize(boost::multi_array<T, 2, A> & v) {
//     v.resize(boost::extents[MATRIX_SIZE][MATRIX_SIZE]);
//     v = creator<boost::multi_array<T, 2, A> >::random();
// }
// template<typename T,  typename A> void initialize(alps::multi_array<T, 2, A> & v) {
//     v.resize(boost::extents[MATRIX_SIZE][MATRIX_SIZE]);
//     v = creator<alps::multi_array<T, 2, A> >::random();
// }
// template<typename T, typename A> void initialize(boost::multi_array<T, 3, A> & v) {
//     v.resize(boost::extents[MATRIX_SIZE][MATRIX_SIZE][MATRIX_SIZE]);
//     v = creator<boost::multi_array<T, 3, A> >::random();
// }
// template<typename T,  typename A> void initialize(alps::multi_array<T, 3, A> & v) {
//     v.resize(boost::extents[MATRIX_SIZE][MATRIX_SIZE][MATRIX_SIZE]);
//     v = creator<alps::multi_array<T, 3, A> >::random();
// }
// template<typename T, typename A> void initialize(boost::multi_array<T, 4, A> & v) {
//     v.resize(boost::extents[MATRIX_SIZE][MATRIX_SIZE][MATRIX_SIZE][MATRIX_SIZE]);
//     v = creator<boost::multi_array<T, 4, A> >::random();
// }
// template<typename T,  typename A> void initialize(alps::multi_array<T, 4, A> & v) {
//     v.resize(boost::extents[MATRIX_SIZE][MATRIX_SIZE][MATRIX_SIZE][MATRIX_SIZE]);
//     v = creator<alps::multi_array<T, 4, A> >::random();
// }

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
            return compare(v, std::integral_constant<bool,
                  (std::is_same<T, double>::value || std::is_same<T, float>::value) &&
                  (std::is_same<U, double>::value || std::is_same<U, float>::value)
            >());
        }
    private:
        bool compare(cast_type<T, U> const & v, std::true_type) const {
            U diff = (base_type::has_u ? base_type::u : alps::cast<U>(base_type::t)) - (v.has_u ? v.u : alps::cast<U>(v.t));
            return (diff > 0 ? diff : -diff) / ((base_type::has_u ? base_type::u : alps::cast<U>(base_type::t)) + (v.has_u ? v.u : alps::cast<U>(v.t))) / 2 < 1e-4;
        }
        bool compare(cast_type<T, U> const & v, std::false_type) const {
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
            , std::vector<std::size_t> /*size*/ = std::vector<std::size_t>()
            , std::vector<std::size_t> /*chunk*/ = std::vector<std::size_t>()
            , std::vector<std::size_t> /*offset*/ = std::vector<std::size_t>()
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
            , std::vector<std::size_t> /*chunk*/ = std::vector<std::size_t>()
            , std::vector<std::size_t> /*offset*/ = std::vector<std::size_t>()
        ) {
            std::string s;
            ar >> alps::make_pvp(path, s);
            value = (s == "plus" ? PLUS : MINUS);
        }

        template<> struct scalar_type<enum_vec_type> {
            typedef boost::int_t<sizeof(enum_vec_type) * 8>::exact type;
        };

        template<> struct is_continuous<enum_vec_type> : public std::true_type {};

        template<> struct has_complex_elements<enum_vec_type> : public std::false_type {};

        namespace detail {

            template<> struct get_extent<enum_vec_type> {
                static std::vector<std::size_t> apply(enum_vec_type const & /*value*/) {
                    return std::vector<std::size_t>();
                }
            };

            template<> struct set_extent<enum_vec_type> {
                static void apply(enum_vec_type &, std::vector<std::size_t> const &) {}
            };

            template<> struct is_vectorizable<enum_vec_type> {
                static bool apply(enum_vec_type const & /*value*/) {
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
    static void apply(T & /*value*/) {}
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
                if (std::is_scalar<T>::value)                                                    \
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
                    if (std::is_scalar<T>::value)                                                \
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
                        if (std::is_scalar<T>::value)                                            \
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
                            if (std::is_scalar<T>::value)                                        \
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
}/*                                                                                                  \
template<typename T> struct creator< C < alps::numeric::matrix<T> > > {                            \
    typedef C < alps::numeric::matrix<T> > base_type;                                              \
    static base_type random() {                                                                    \
        base_type value(                                                                           \
            VECTOR_SIZE, alps::numeric::matrix<T>(MATRIX_SIZE, MATRIX_SIZE)                        \
        );                                                                                         \
        for (std::size_t i = 0; i < VECTOR_SIZE; ++i)                                              \
            for (std::size_t j = 0; j < MATRIX_SIZE; ++j)                                          \
                for (std::size_t k = 0; k < MATRIX_SIZE; ++k)                                      \
                    if (std::is_scalar<T>::value)                                                \
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
};*/                                                                                                 \
HDF5_DEFINE_MULTI_ARRAY_TYPE(boost, C)/*                                                             \
HDF5_DEFINE_MULTI_ARRAY_TYPE(alps, C)*/

HDF5_DEFINE_VECTOR_TYPE(std::vector)
HDF5_DEFINE_VECTOR_TYPE(std::valarray)
HDF5_DEFINE_VECTOR_TYPE(std::deque)
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
            initialize(const_cast<typename boost::remove_const<T>::type &>(value.first[i]));
        return value;
    }
    static base_type empty() {
        return std::make_pair(static_cast<typename boost::remove_const<T>::type *>(NULL), std::vector<U>());
    }
    static base_type special() {
        base_type value = std::make_pair(new typename boost::remove_const<T>::type[MATRIX_SIZE * MATRIX_SIZE * MATRIX_SIZE], std::vector<U>(3, MATRIX_SIZE));
        for (std::size_t i = 0; i < MATRIX_SIZE * MATRIX_SIZE * MATRIX_SIZE; ++i)
            initialize(const_cast<typename boost::remove_const<T>::type &>(value.first[i]));
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
                if (std::is_scalar<T>::value)                                                                                                     \
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
                    if (std::is_scalar<T>::value)                                                                                                 \
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
                        if (std::is_scalar<T>::value)                                                                                             \
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
                            if (std::is_scalar<T>::value)                                                                                         \
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
// HDF5_DEFINE_MULTI_ARRAY_TYPE(alps)


// template<typename T> struct creator<alps::numeric::matrix<T> > {
//     typedef alps::numeric::matrix<T> base_type;
//     static base_type random() {
//         base_type value (MATRIX_SIZE, MATRIX_SIZE);
//         for (std::size_t i = 0; i < MATRIX_SIZE; ++i)
//             for (std::size_t j = 0; j < MATRIX_SIZE; ++j)
//                 if (std::is_scalar<T>::value)
//                     initialize(value(i, j));
//                 else
//                     value(i, j) = creator<T>::random();
//         return value;
//     }
//     static base_type empty() { return base_type(); }
//     static base_type special() { return base_type(); }
//     template<typename X> static base_type random(X const &) { return base_type(MATRIX_SIZE, MATRIX_SIZE); }
//     template<typename X> static base_type empty(X const &) { return base_type(); }
//     template<typename X> static base_type special(X const &) { return base_type(); }
// };
// template<typename T> bool equal(alps::numeric::matrix<T> const & a, alps::numeric::matrix<T> const & b) {
//     return a == b;
// }

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
HDF5_DEFINE_VECTOR_VECTOR_TYPE(std::deque, std::deque)
HDF5_DEFINE_VECTOR_VECTOR_TYPE(std::deque, std::vector)
HDF5_DEFINE_VECTOR_VECTOR_TYPE(std::vector, std::deque)
HDF5_DEFINE_VECTOR_VECTOR_TYPE(std::valarray, std::deque)
HDF5_DEFINE_VECTOR_VECTOR_TYPE(std::deque, std::valarray)
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

template<typename base_type> struct hdf5_test {
    static bool write(std::string const & filename, std::true_type) {
        std::vector<std::size_t> size_0;
        base_type* write_0_value = NULL;
        std::size_t length = MATRIX_SIZE;
        std::vector<std::size_t> size_1(1, MATRIX_SIZE);
        base_type write_1_value[MATRIX_SIZE];
        std::vector<std::size_t> size_2(2, MATRIX_SIZE);
        base_type write_2_value[MATRIX_SIZE][MATRIX_SIZE];
        std::vector<std::size_t> size_3(3, MATRIX_SIZE);
        base_type write_3_value[MATRIX_SIZE][MATRIX_SIZE][MATRIX_SIZE];
        for (std::size_t i = 0; i < MATRIX_SIZE; ++i) {
            initialize(write_1_value[i]);
            for (std::size_t j = 0; j < MATRIX_SIZE; ++j) {
                initialize(write_2_value[i][j]);
                for (std::size_t k = 0; k < MATRIX_SIZE; ++k)
                    initialize(write_3_value[i][j][k]);
            }
        }
        {
            alps::hdf5::archive oar(filename, SZIP_COMPRESS ? "ca" : "a");
            if (IS_ATTRIBUTE)
                oar["/data"] << 0;
            oar
                << alps::make_pvp(std::string(IS_ATTRIBUTE ? "/data/@" : "/") + "len", &write_1_value[0], length)
                << alps::make_pvp(std::string(IS_ATTRIBUTE ? "/data/@" : "/") + "len_0", write_0_value, 0)
                << alps::make_pvp(std::string(IS_ATTRIBUTE ? "/data/@" : "/") + "ptr_0", write_0_value, size_0)
                << alps::make_pvp(std::string(IS_ATTRIBUTE ? "/data/@" : "/") + "ptr_1", &write_1_value[0], size_1)
                << alps::make_pvp(std::string(IS_ATTRIBUTE ? "/data/@" : "/") + "ptr_2", &write_2_value[0][0], size_2)
                << alps::make_pvp(std::string(IS_ATTRIBUTE ? "/data/@" : "/") + "ptr_3", &write_3_value[0][0][0], size_3)
            ;
        }
        {
            base_type* read_0_value = NULL;
            base_type read_1_len_value[MATRIX_SIZE], read_1_value[MATRIX_SIZE];
            base_type read_2_value[MATRIX_SIZE][MATRIX_SIZE];
            base_type read_3_value[MATRIX_SIZE][MATRIX_SIZE][MATRIX_SIZE];
            alps::hdf5::archive iar(filename, SZIP_COMPRESS ? "rc" : "r");
            iar
                >> alps::make_pvp(std::string(IS_ATTRIBUTE ? "/data/@" : "/") + "len", &read_1_len_value[0], length)
                >> alps::make_pvp(std::string(IS_ATTRIBUTE ? "/data/@" : "/") + "len_0", write_0_value, 0)
                >> alps::make_pvp(std::string(IS_ATTRIBUTE ? "/data/@" : "/") + "ptr_0", read_0_value, size_0)
                >> alps::make_pvp(std::string(IS_ATTRIBUTE ? "/data/@" : "/") + "ptr_1", &read_1_value[0], size_1)
                >> alps::make_pvp(std::string(IS_ATTRIBUTE ? "/data/@" : "/") + "ptr_2", &read_2_value[0][0], size_2)
                >> alps::make_pvp(std::string(IS_ATTRIBUTE ? "/data/@" : "/") + "ptr_3", &read_3_value[0][0][0], size_3)
            ;
            return write_0_value == read_0_value
                && equal(&write_1_value[0], &read_1_len_value[0], length)
                && equal(&write_1_value[0], &read_1_value[0], size_1[0])
                && equal(&write_2_value[0][0], &read_2_value[0][0], size_2[0] * size_2[1])
                && equal(&write_3_value[0][0][0], &read_3_value[0][0][0], size_3[0] * size_3[1] * size_3[2])
            ;
        }
    }
    static bool write(std::string const & filename, std::false_type) {
        base_type random_write(creator<base_type>::random());
        base_type empty_write(creator<base_type>::empty());
        base_type special_write(creator<base_type>::special());
        bool result;
        {
            alps::hdf5::archive oar(filename, SZIP_COMPRESS ? "ca" : "a");
            if (IS_ATTRIBUTE)
                oar["/data"] << 0;
            oar[std::string(IS_ATTRIBUTE ? "/data/@" : "/") + "random"] << random_write;
            oar[std::string(IS_ATTRIBUTE ? "/data/@" : "/") + "empty"]<< empty_write;
            if (!IS_ATTRIBUTE)
                oar["/special"] << special_write;
        }
        {
            alps::hdf5::archive iar(filename);
            base_type random_read(creator<base_type>::random(iar));
            base_type empty_read(creator<base_type>::empty(iar));
            base_type special_read(creator<base_type>::special(iar));
            iar[std::string(IS_ATTRIBUTE ? "/data/@" : "/") + "random"] >> random_read;
            iar[std::string(IS_ATTRIBUTE ? "/data/@" : "/") + "empty"] >> empty_read;
            if (!IS_ATTRIBUTE)
                iar["/special"] >> special_read;
            result = equal(random_write, random_read) && equal(empty_write, empty_read) && (IS_ATTRIBUTE || equal(special_write, special_read));
            destructor<base_type>::apply(random_read);
            destructor<base_type>::apply(empty_read);
            if (!IS_ATTRIBUTE)
                destructor<base_type>::apply(special_read);
        }
        destructor<base_type>::apply(random_write);
        destructor<base_type>::apply(empty_write);
        if (!IS_ATTRIBUTE)
            destructor<base_type>::apply(special_write);
        return result;
    }
    template<typename data_type> static bool overwrite_helper(std::string const & filename) {
        data_type random_write(creator<data_type>::random());
        bool result;
        alps::hdf5::archive ar(filename, SZIP_COMPRESS ? "ca" : "a");
        {
            if (IS_ATTRIBUTE)
                ar["/data"] << 0;
            ar[std::string(IS_ATTRIBUTE ? "/data/@" : "/") + "overwrite"] << random_write;
        }
        {
            data_type random_read(creator<data_type>::random(ar));
            ar[std::string(IS_ATTRIBUTE ? "/data/@" : "/") + "overwrite"] >> random_read;
            result = equal(random_write, random_read);
            destructor<data_type>::apply(random_read);
        }
        destructor<data_type>::apply(random_write);
        return result;
    }
    template<typename unused> static bool overwrite(std::string const & filename, unused) {
        return overwrite_helper<int>(filename);
        return overwrite_helper<base_type>(filename);
        return overwrite_helper<std::complex<double> >(filename);
        return overwrite_helper<base_type>(filename);
        return overwrite_helper<double>(filename);
        return overwrite_helper<base_type>(filename);
        return overwrite_helper<std::vector<double> >(filename);
        return overwrite_helper<base_type>(filename);
        return overwrite_helper<std::string>(filename);
        return overwrite_helper<base_type>(filename);
    }
};

template<typename T> struct hdf5_test<boost::shared_array<T> > {
    static bool write(std::string const & filename, std::false_type) {
        std::size_t length = MATRIX_SIZE;
        std::vector<std::size_t> size_1(1, MATRIX_SIZE);
        boost::shared_array<T> write_1_value(new T[MATRIX_SIZE]);
        std::vector<std::size_t> size_2(2, MATRIX_SIZE);
        boost::shared_array<T> write_2_value(new T[MATRIX_SIZE * MATRIX_SIZE]);
        std::vector<std::size_t> size_3(3, MATRIX_SIZE);
        boost::shared_array<T> write_3_value(new T[MATRIX_SIZE * MATRIX_SIZE * MATRIX_SIZE]);
        for (std::size_t i = 0; i < MATRIX_SIZE; ++i) {
            initialize(write_1_value[i]);
            for (std::size_t j = 0; j < MATRIX_SIZE; ++j) {
                initialize(write_2_value[i * MATRIX_SIZE + j]);
                for (std::size_t k = 0; k < MATRIX_SIZE; ++k)
                    initialize(write_3_value[(i * MATRIX_SIZE + j) * MATRIX_SIZE + k]);
            }
        }
        {
            alps::hdf5::archive oar(filename, SZIP_COMPRESS ? "ca" : "a");
            if (IS_ATTRIBUTE)
                oar["/data"] << 0;
            oar
                << alps::make_pvp(std::string(IS_ATTRIBUTE ? "/data/@" : "/") + "len", write_1_value, length)
                << alps::make_pvp(std::string(IS_ATTRIBUTE ? "/data/@" : "/") + "ptr_1", write_1_value, size_1)
                << alps::make_pvp(std::string(IS_ATTRIBUTE ? "/data/@" : "/") + "ptr_2", write_2_value, size_2)
                << alps::make_pvp(std::string(IS_ATTRIBUTE ? "/data/@" : "/") + "ptr_3", write_3_value, size_3)
            ;
        }
        {
            boost::shared_array<T> read_1_len_value(new T[MATRIX_SIZE]), read_1_value(new T[MATRIX_SIZE]);
            boost::shared_array<T> read_2_value(new T[MATRIX_SIZE * MATRIX_SIZE]);
            boost::shared_array<T> read_3_value(new T[MATRIX_SIZE * MATRIX_SIZE * MATRIX_SIZE]);
            alps::hdf5::archive iar(filename);
            iar
                >> alps::make_pvp(std::string(IS_ATTRIBUTE ? "/data/@" : "/") + "len", read_1_len_value, length)
                >> alps::make_pvp(std::string(IS_ATTRIBUTE ? "/data/@" : "/") + "ptr_1", read_1_value, size_1)
                >> alps::make_pvp(std::string(IS_ATTRIBUTE ? "/data/@" : "/") + "ptr_2", read_2_value, size_2)
                >> alps::make_pvp(std::string(IS_ATTRIBUTE ? "/data/@" : "/") + "ptr_3", read_3_value, size_3)
            ;
            return equal(write_1_value.get(), read_1_len_value.get(), length)
                && equal(write_1_value.get(), read_1_value.get(), size_1[0])
                && equal(write_2_value.get(), read_2_value.get(), size_2[0] * size_2[1])
                && equal(write_3_value.get(), read_3_value.get(), size_3[0] * size_3[1] * size_3[2])
            ;
        }
    }
    static bool overwrite(std::string const & filename, std::false_type) {
        // TODO: implement test for write type A and overwrite with type B
        return true;
    }
};

// TODO: this should be possible
template<typename T> struct skip_attribute: public std::false_type {};

template<typename T> struct skip_attribute<userdefined_class<T> >: public std::true_type {};
template<typename T, typename U> struct skip_attribute<cast_type<T, U> >: public std::true_type {};
template<> struct skip_attribute<enum_type>: public std::true_type {};

template<> struct skip_attribute<std::vector<bool> >: public std::true_type {};
template<typename T> struct skip_attribute<std::vector<std::vector<T> > >: public std::true_type {};
template<typename T> struct skip_attribute<std::valarray<std::vector<T> > >: public std::true_type {};
template<typename T> struct skip_attribute<std::vector<std::valarray<T> > >: public std::true_type {};
template<typename T> struct skip_attribute<std::valarray<std::valarray<T> > >: public std::true_type {};

template<typename T, std::size_t N> struct skip_attribute<boost::array<std::vector<T>, N> >: public std::true_type {};

template<typename T, std::size_t N, typename A> struct skip_attribute<std::vector<boost::multi_array<T, N, A> > >: public std::true_type {};
// template<typename T, std::size_t N, typename A> struct skip_attribute<std::vector<alps::multi_array<T, N, A> > >: public std::true_type {};

template<typename T, std::size_t N, typename A> struct skip_attribute<boost::multi_array<T, N, A> * >: public std::true_type {};
// template<typename T, std::size_t N, typename A> struct skip_attribute<alps::multi_array<T, N, A> * >: public std::true_type {};

template <
    typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9
> struct skip_attribute<std::vector<boost::tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> > >: public std::true_type {};

// template <typename T, typename M> struct skip_attribute<alps::numeric::matrix<T, M> > : public std::true_type {};

template<typename T> struct skip_attribute<T *>: public skip_attribute<T> {};
template<typename T> struct skip_attribute<std::vector<T> >: public skip_attribute<T> {};
template<typename T> struct skip_attribute<std::valarray<T> >: public skip_attribute<T> {};
template<typename T, typename U> struct skip_attribute<std::pair<T, U> >:
    public std::integral_constant<bool, skip_attribute<T>::value || skip_attribute<U>::value > {};

template<typename T> struct skip_attribute<std::pair<T *, std::vector<std::size_t> > >: public skip_attribute<T> {};
template<typename T> struct skip_attribute<std::pair<std::vector<T> *, std::vector<std::size_t> > >: public std::true_type {};
template<typename T, std::size_t N, typename A> struct skip_attribute<std::pair<boost::multi_array<T, N, A> *, std::vector<std::size_t> > >: public std::true_type {};
// template<typename T, std::size_t N, typename A> struct skip_attribute<std::pair<alps::multi_array<T, N, A> *, std::vector<std::size_t> > >: public std::true_type {};

template<typename T> struct skip_attribute<boost::shared_array<T> >: public skip_attribute<T> {};

#include "gtest/gtest.h"

template<typename XXXX> class TypedTestEncapsulation: public ::testing::Test{
public:
  TypedTestEncapsulation(){
    alps::testing::unique_file ufile("hdf5_io_generic_test.h5.", alps::testing::unique_file::REMOVE_NOW);
    const std::string& filename = ufile.name();

    result_ = true;
    if (IS_ATTRIBUTE && skip_attribute<XXXX >::value)
      std::cout << "SKIP" << std::endl;
    else {
      for (std::size_t i = 32; i && result_; --i){
        result_=hdf5_test<typename boost::remove_pointer<XXXX>::type >::write(filename, typename std::is_pointer< XXXX >::type());
        EXPECT_TRUE(result_);
      }
      {
        alps::hdf5::archive iar1(filename, SZIP_COMPRESS ? "ca" : "a");
        alps::hdf5::archive iar2(filename, SZIP_COMPRESS ? "ca" : "a");
        alps::hdf5::archive iar3 = iar1;
        for (std::size_t i = 32; i && result_; --i){
          result_=hdf5_test<typename boost::remove_pointer< XXXX >::type >::overwrite(filename, typename std::is_pointer< XXXX >::type());
          EXPECT_TRUE(result_);
        }
      }
      //std::cout << (result_ ? "SUCCESS" : "FAILURE") << std::endl;
    }
  }
  bool result_;
};

template<typename TYPE> class ScalarTypedTestEncapsulation: public TypedTestEncapsulation<TYPE>{};
template<typename TYPE> class VectorTypedTestEncapsulation: public TypedTestEncapsulation<TYPE>{};
template<typename TYPE> class ValarrayTypedTestEncapsulation: public TypedTestEncapsulation<TYPE>{};
template<typename TYPE> class PairTypedTestEncapsulation: public TypedTestEncapsulation<TYPE>{};
template<typename TYPE> class VectorVectorTypedTestEncapsulation: public TypedTestEncapsulation<TYPE>{};
template<typename TYPE> class VectorValarrayTypedTestEncapsulation: public TypedTestEncapsulation<TYPE>{};
template<typename TYPE> class PairVectorTypedTestEncapsulation: public TypedTestEncapsulation<TYPE>{};
template<typename TYPE> class EnumTypedTestEncapsulation: public TypedTestEncapsulation<TYPE>{};
template<typename TYPE> class UserDefinedTypedTestEncapsulation: public TypedTestEncapsulation<TYPE>{};
template<typename TYPE> class CastTypedTestEncapsulation: public TypedTestEncapsulation<TYPE>{};
template<typename TYPE> class PointerTypedTestEncapsulation: public TypedTestEncapsulation<TYPE>{};
template<typename TYPE> class RemainingTypedTestEncapsulation: public TypedTestEncapsulation<TYPE>{};


typedef ::testing::Types<
bool, int, short, long, float, double,
std::size_t, std::string, std::complex<float>, std::complex<double>, std::complex<long double>,
boost::int8_t, boost::uint8_t, boost::int16_t, boost::uint16_t, boost::int32_t, boost::uint32_t, boost::int64_t, boost::uint64_t> hdf5ScalarTypes;

typedef ::testing::Types<std::vector<bool>,
std::pair<std::vector<bool> *, std::vector<std::size_t> >,
std::vector<std::size_t>, std::vector<short>, std::vector<int>, std::vector<long>, std::vector<float>, std::vector<double>, std::vector<std::complex<double> >, std::vector<std::string> >hdf5VectorTypes;

typedef ::testing::Types<
std::valarray<int>, std::valarray<double>, std::valarray<std::complex<double> > > hdf5ValarrayTypes;

typedef ::testing::Types<
std::pair<int *, std::vector<std::size_t> >,
std::pair<double *, std::vector<std::size_t> >,
std::pair<std::complex<double> *,std::vector<std::size_t> >,
std::pair<std::string *,std::vector<std::size_t> > > hdf5PairTypes;

typedef ::testing::Types<
std::vector<std::vector<int> >,
std::vector<std::vector<double> >,
std::vector<std::vector<std::complex<double> > >,
std::vector<std::vector<std::string> >,
std::vector<std::vector<std::vector<int> > >,
std::vector<std::vector<std::vector<double> > >,
std::vector<std::vector<std::vector<std::complex<double> > > >,
std::vector<std::vector<std::vector<std::string> > > > hdf5VectorVectorTypes;


typedef ::testing::Types<
std::vector<std::valarray<int> >,
std::valarray<std::vector<double> > >hdf5VectorValarrayTypes;

typedef ::testing::Types<
std::pair<std::vector<int> *, std::vector<std::size_t> >,
std::pair<std::vector<double> *, std::vector<std::size_t> >,
std::pair<std::vector<std::complex<double> > *, std::vector<std::size_t> >,
std::pair<std::vector<std::string> *, std::vector<std::size_t> > >hdf5PairVectorTypes;

typedef ::testing::Types<
enum_type,
std::vector<enum_type>,
std::vector<std::vector<enum_type> >,
std::pair<enum_type *, std::vector<std::size_t> >,
std::vector<std::valarray<enum_type> >,
std::pair<std::vector<enum_type> *, std::vector<std::size_t> >,
std::pair<std::vector<std::vector<enum_type> > *, std::vector<std::size_t> >,
enum_vec_type,
std::vector<enum_vec_type>,
std::vector<std::vector<enum_vec_type> >,
std::pair<enum_vec_type *, std::vector<std::size_t> >,
std::vector<std::valarray<enum_vec_type> >,
std::pair<std::vector<enum_vec_type> *,
std::vector<std::size_t> >,
std::pair<std::vector<std::vector<enum_vec_type> > *, std::vector<std::size_t> > >hdf5EnumTypes;


typedef ::testing::Types<
 userdefined_class<std::size_t>,
userdefined_class<short>,
userdefined_class<int>,
userdefined_class<long>,
userdefined_class<float>,
userdefined_class<double>,
userdefined_class<std::complex<double> >,
userdefined_class<std::string>,
std::vector<userdefined_class<double> >,
std::vector<std::vector<userdefined_class<double> > >,
std::pair<userdefined_class<double> *, std::vector<std::size_t> >  > hdf5UserDefinedTypes;

typedef ::testing::Types<
cast_type<int, long>,
cast_type<int, double>,
cast_type<double, std::string>,
cast_type<int, std::string>,
cast_type<float, double>,
cast_type<short, float>,
std::vector<cast_type<int, double> >,
std::vector<std::vector<cast_type<int, double> > >,
std::pair<cast_type<int, double> *, std::vector<std::size_t> >,
std::vector<std::valarray<cast_type<int, double> > >,
std::vector<cast_type<double, std::string> >,
std::vector<std::vector<cast_type<double, std::string> > >,
std::pair<cast_type<double, std::string> *, std::vector<std::size_t> > > hdf5CastTypes;

typedef ::testing::Types<
int *,short *,long *,float *,double *,
std::size_t *,std::string *,std::complex<double> *,
enum_type *,enum_vec_type *,userdefined_class<double> *,cast_type<int, double> *,cast_type<int, std::string> * >hdf5PointerTypes;

typedef ::testing::Types<
boost::shared_array<int>,boost::shared_array<short>,boost::shared_array<long>,boost::shared_array<float>,boost::shared_array<double>, boost::shared_array<std::size_t>,boost::shared_array<std::string>,boost::shared_array<std::complex<double> >,boost::shared_array<enum_type>,boost::shared_array<enum_vec_type>,boost::shared_array<userdefined_class<double> >,boost::shared_array<cast_type<int, double> >,boost::shared_array<cast_type<int, std::string> >,cast_type<std::vector<int>, std::valarray<int> >,std::pair<double, int>,std::pair<double, std::complex<double> >,std::pair<cast_type<int, std::string>, enum_type>,std::pair<enum_type, cast_type<int, double> >,std::pair<std::vector<cast_type<int, std::string> >, std::pair<double, int> >,std::pair<std::pair<std::vector<enum_type> *, std::vector<std::size_t> >, enum_type>,cast_type<std::valarray<int>, std::vector<int> >,cast_type<std::pair<int *, std::vector<std::size_t> >, std::vector<std::vector<std::vector<int> > > >,cast_type<std::pair<int *, std::vector<std::size_t> >, std::vector<std::vector<std::vector<double> > > >,std::pair<cast_type<std::vector<int>, std::valarray<long> > *, std::vector<std::size_t> >,cast_type<std::vector<int>, std::valarray<double> >,std::vector<std::size_t, std::allocator<std::size_t> >,std::vector<short, std::allocator<short> >,std::vector<int, std::allocator<int> >,std::vector<long, std::allocator<long> >,std::vector<float, std::allocator<float> >,std::vector<double, std::allocator<double> >,std::vector<std::complex<double>, std::allocator<std::complex<double> > >,std::vector<std::string, std::allocator<std::string> >,std::vector<std::vector<int, std::allocator<int> > >,std::vector<std::vector<double>, std::allocator<std::vector<double> > >,std::vector<std::vector<std::complex<double>, std::allocator<std::complex<double> > >, std::allocator<std::vector<std::complex<double>, std::allocator<std::complex<double> > > > >,std::vector<std::vector<std::string, std::allocator<std::string> >, std::allocator<std::vector<std::string, std::allocator<std::string> > > >,boost::array<int, 20>,boost::array<long double, 20>,boost::array<float, 20>,boost::array<unsigned long long, 20>,boost::array<boost::array<std::complex<double>, 20>, 20>,std::vector<boost::array<int, 4> >,boost::array<std::vector<int>, 4>,std::vector<boost::array<std::vector<int>, 4> >,boost::tuple<int, double, float, std::complex<double> >,std::vector<boost::tuple<char, bool, long long> > > hdf5RemainingTypes;

TYPED_TEST_CASE(ScalarTypedTestEncapsulation, hdf5ScalarTypes);
TYPED_TEST(ScalarTypedTestEncapsulation, TestTypes) {
  EXPECT_TRUE(this->result_);
}
#ifdef ExtensiveTesting
TYPED_TEST_CASE(VectorTypedTestEncapsulation, hdf5ScalarTypes);
TYPED_TEST(VectorTypedTestEncapsulation, TestTypes) {
  EXPECT_TRUE(this->result_);
}
TYPED_TEST_CASE(ValarrayTypedTestEncapsulation, hdf5ValarrayTypes);
TYPED_TEST(ValarrayTypedTestEncapsulation, TestTypes) {
  EXPECT_TRUE(this->result_);
}
TYPED_TEST_CASE(PairTypedTestEncapsulation, hdf5PairTypes);
TYPED_TEST(PairTypedTestEncapsulation, TestTypes) {
  EXPECT_TRUE(this->result_);
}
TYPED_TEST_CASE(VectorVectorTypedTestEncapsulation, hdf5VectorVectorTypes);
TYPED_TEST(VectorVectorTypedTestEncapsulation, TestTypes) {
  EXPECT_TRUE(this->result_);
}
TYPED_TEST_CASE(VectorValarrayTypedTestEncapsulation, hdf5VectorValarrayTypes);
TYPED_TEST(VectorValarrayTypedTestEncapsulation, TestTypes) {
  EXPECT_TRUE(this->result_);
}
TYPED_TEST_CASE(PairVectorTypedTestEncapsulation, hdf5PairVectorTypes);
TYPED_TEST(PairVectorTypedTestEncapsulation, TestTypes) {
  EXPECT_TRUE(this->result_);
}
TYPED_TEST_CASE(EnumTypedTestEncapsulation, hdf5EnumTypes);
TYPED_TEST(EnumTypedTestEncapsulation, TestTypes) {
  EXPECT_TRUE(this->result_);
}
TYPED_TEST_CASE(UserDefinedTypedTestEncapsulation, hdf5UserDefinedTypes);
TYPED_TEST(UserDefinedTypedTestEncapsulation, TestTypes) {
  EXPECT_TRUE(this->result_);
}
TYPED_TEST_CASE(CastTypedTestEncapsulation, hdf5CastTypes);
TYPED_TEST(CastTypedTestEncapsulation, TestTypes) {
  EXPECT_TRUE(this->result_);
}
TYPED_TEST_CASE(PointerTypedTestEncapsulation, hdf5PointerTypes);
TYPED_TEST(PointerTypedTestEncapsulation, TestTypes) {
  EXPECT_TRUE(this->result_);
}
TYPED_TEST_CASE(RemainingTypedTestEncapsulation, hdf5RemainingTypes);
TYPED_TEST(RemainingTypedTestEncapsulation, TestTypes) {
  EXPECT_TRUE(this->result_);
}
#endif



// int main(int argc, char **argv)
// {
//     ::testing::InitGoogleTest(&argc, argv);
//     return RUN_ALL_TESTS();
// }




