/*
 * Copyright (C) 1998-2015 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_ACCUMULATOR_ERROR_HPP
#define ALPS_ACCUMULATOR_ERROR_HPP

#include <alps/config.hpp>

#include <alps/accumulators/feature.hpp>
#include <alps/accumulators/parameter.hpp>
#include <alps/accumulators/feature/mean.hpp>
#include <alps/accumulators/feature/count.hpp>

#include <alps/hdf5/archive.hpp>
#include <alps/numeric/inf.hpp>
#include <alps/numeric/check_size.hpp>
#include <alps/numeric/vector_functions.hpp>
#include <alps/numeric/boost_array_functions.hpp>
 
#include <alps/utilities/stacktrace.hpp>
#include <alps/utilities/short_print.hpp>

#include <boost/mpl/if.hpp>
#include <boost/utility.hpp>
#include <boost/type_traits/is_scalar.hpp>
#include <boost/type_traits/is_integral.hpp>

#include <stdexcept>

namespace alps {
    namespace accumulators {
        // this should be called namespace tag { struct error; }
        // but gcc <= 4.4 has lookup error, so name it different
        struct error_tag;

        template<typename T> struct error_type : public mean_type<T> {};

        template<typename T> struct has_feature<T, error_tag> {
            template<typename R, typename C> static char helper(R(C::*)() const);
            template<typename R, typename C> static char helper(R(C::*)(std::size_t) const);
            template<typename C> static char check(boost::integral_constant<std::size_t, sizeof(helper(&C::error))>*);
            template<typename C> static double check(...);
            typedef boost::integral_constant<bool, sizeof(char) == sizeof(check<T>(0))> type;
        };

        template<typename T> typename error_type<T>::type error(T const & arg) {
            return arg.error();
        }

        namespace detail {

            template<typename A> typename boost::enable_if<
                  typename has_feature<A, error_tag>::type
                , typename error_type<A>::type
            >::type error_impl(A const & acc) {
                return error(acc);
            }

            template<typename A> typename boost::disable_if<
                  typename has_feature<A, error_tag>::type
                , typename error_type<A>::type
            >::type error_impl(A const & acc) {
                throw std::runtime_error(std::string(typeid(A).name()) + " has no error-method" + ALPS_STACKTRACE);
                return typename error_type<A>::type();
            }
        }

        namespace impl {

            template<typename T, typename B> struct Accumulator<T, error_tag, B> : public B {

                public:
                    typedef typename alps::accumulators::error_type<B>::type error_type;
                    typedef typename alps::hdf5::scalar_type<error_type>::type error_scalar_type;
                    typedef Result<T, error_tag, typename B::result_type> result_type;

                    Accumulator(): B(), m_sum2(T()) {}

                    Accumulator(Accumulator const & arg): B(arg), m_sum2(arg.m_sum2) {}

                    template<typename ArgumentPack> Accumulator(ArgumentPack const & args, typename boost::disable_if<is_accumulator<ArgumentPack>, int>::type = 0)
                        : B(args), m_sum2(T())
                    {}

                    error_type const error() const {
                        using std::sqrt;
                        using alps::numeric::sqrt;
                        using alps::numeric::operator/;
                        using alps::numeric::operator-;
                        using alps::numeric::operator*;

                        // TODO: make library for scalar type
                        error_scalar_type cnt = B::count();
                        if (cnt<=1) return alps::numeric::inf<error_type>(m_sum2);
                        return sqrt((m_sum2 / cnt - B::mean() * B::mean()) / (cnt - 1));
                    }

                    using B::operator();
                    void operator()(T const & val) {
                        using alps::numeric::operator*;
                        using alps::numeric::operator+=;
                        using alps::numeric::check_size;

                        B::operator()(val);
                        check_size(m_sum2, val);
                        m_sum2 += val * val;
                    }

                    template<typename S> void print(S & os) const {
                        B::print(os);
                        os << " +/-" << alps::short_print(error());
                    }

                    void save(hdf5::archive & ar) const {
                        B::save(ar);
                        ar["mean/error"] = error();
                    }

                    void load(hdf5::archive & ar) { // TODO: make archive const
                        using alps::numeric::operator*;
                        using alps::numeric::operator+;

                        B::load(ar);
                        error_type error;
                        ar["mean/error"] >> error;
                        // TODO: make library for scalar type
                        error_scalar_type cnt = B::count();
                        m_sum2 = (error * error * (cnt - 1) + B::mean() * B::mean()) * cnt;
                    }

                    static std::size_t rank() { return B::rank() + 1; }
                    static bool can_load(hdf5::archive & ar) { // TODO: make archive const
                        using alps::hdf5::get_extent;
                        const char name[]="mean/error";
                        return B::can_load(ar)
                            && ar.is_data(name) 
                            && ar.is_datatype<error_scalar_type>(name)
                            && boost::is_scalar<T>::value == ar.is_scalar(name)
                            && (boost::is_scalar<T>::value || get_extent(T()).size() == ar.dimensions(name))
                        ;
                    }

                    void reset() {
                        B::reset();
                        m_sum2 = T();
                    }

                    /// Merge the mean & error of given accumulator of type A into this accumulator  @param rhs Accumulator to merge 
                    template <typename A>
                    void merge(const A& rhs)
                    {
                      using alps::numeric::operator+=;
                      using alps::numeric::check_size;
                      B::merge(rhs);
                      check_size(m_sum2, rhs.m_sum2);
                      m_sum2 += rhs.m_sum2;
                    }

#ifdef ALPS_HAVE_MPI
                    void collective_merge(
                          boost::mpi::communicator const & comm
                        , int root
                    ) {
                        if (comm.rank() == root) {
                            B::collective_merge(comm, root);
                            B::reduce_if(comm, T(m_sum2), m_sum2, std::plus<typename alps::hdf5::scalar_type<T>::type>(), root);
                        } else
                            const_cast<Accumulator<T, error_tag, B> const *>(this)->collective_merge(comm, root);
                    }

                    void collective_merge(
                          boost::mpi::communicator const & comm
                        , int root
                    ) const {
                        B::collective_merge(comm, root);
                        if (comm.rank() == root)
                            throw std::runtime_error("A const object cannot be root" + ALPS_STACKTRACE);
                        else
                            B::reduce_if(comm, m_sum2, std::plus<typename alps::hdf5::scalar_type<T>::type>(), root);
                    }
#endif

                private:
                    T m_sum2;
            };

            template<typename T, typename B> class Result<T, error_tag, B> : public B {

                public:
                    typedef typename alps::accumulators::error_type<B>::type error_type;
                    typedef typename alps::hdf5::scalar_type<error_type>::type error_scalar_type;
                    template <typename U> struct make_scalar_result_type { typedef void type; };
                    template <typename U> struct make_scalar_result_type< std::vector<U> > { typedef Result<U, error_tag, typename B::scalar_result_type> type; };
                    typedef typename make_scalar_result_type<T>::type scalar_result_type;
                    typedef Result<std::vector<T>, error_tag, typename B::vector_result_type> vector_result_type;

                    Result() 
                        : B()
                        , m_error(error_type()) 
                    {}

                    template<typename A> Result(A const & acc)
                        : B(acc)
                        , m_error(detail::error_impl(acc))
                    {}

                    error_type const error() const { 
                        return m_error; 
                    }

                    template<typename S> void print(S & os) const {
                        B::print(os);
                        os << " +/-" << alps::short_print(error());
                    }

                    void save(hdf5::archive & ar) const {
                        B::save(ar);
                        ar["mean/error"] = error();
                    }

                    void load(hdf5::archive & ar) {
                        B::load(ar);
                        ar["mean/error"] >> m_error;
                    }

                    static std::size_t rank() { return B::rank() + 1; }
                    static bool can_load(hdf5::archive & ar) { // TODO: make archive const
                        using alps::hdf5::get_extent;
                        const char name[]="mean/error";

                        return B::can_load(ar) 
                            && ar.is_data(name) 
                            && ar.is_datatype<error_scalar_type>(name)
                            && boost::is_scalar<T>::value == ar.is_scalar(name)
                            && (boost::is_scalar<T>::value || get_extent(T()).size() == ar.dimensions(name))
                        ;
                    }

                    template<typename U> void operator+=(U const & arg) { augaddsub(arg); B::operator+=(arg); }
                    template<typename U> void operator-=(U const & arg) { augaddsub(arg); B::operator-=(arg); }
                    template<typename U> void operator*=(U const & arg) { augmul(arg); }
                    template<typename U> void operator/=(U const & arg) { augdiv(arg); }
                    void negate() {
                        using alps::numeric::operator-;
                        m_error = -m_error;
                        B::negate();
                    }
                    void inverse() {
                        using alps::numeric::operator*;
                        using alps::numeric::operator/;
                        m_error = this->error() / (this->mean() * this->mean());
                        B::inverse();

                    }

                    #define NUMERIC_FUNCTION_USING                                  \
                        using alps::numeric::sq;                                    \
                        using alps::numeric::cbrt;                                  \
                        using alps::numeric::cb;                                    \
                        using std::sqrt;                                            \
                        using alps::numeric::sqrt;                                  \
                        using std::exp;                                             \
                        using alps::numeric::exp;                                   \
                        using std::log;                                             \
                        using alps::numeric::log;                                   \
                        using std::abs;                                             \
                        using alps::numeric::abs;                                   \
                        using std::pow;                                             \
                        using alps::numeric::pow;                                   \
                        using std::sin;                                             \
                        using alps::numeric::sin;                                   \
                        using std::cos;                                             \
                        using alps::numeric::cos;                                   \
                        using std::tan;                                             \
                        using alps::numeric::tan;                                   \
                        using std::sinh;                                            \
                        using alps::numeric::sinh;                                  \
                        using std::cosh;                                            \
                        using alps::numeric::cosh;                                  \
                        using std::tanh;                                            \
                        using alps::numeric::tanh;                                  \
                        using std::asin;                                            \
                        using alps::numeric::asin;                                  \
                        using std::acos;                                            \
                        using alps::numeric::acos;                                  \
                        using std::atan;                                            \
                        using alps::numeric::atan;                                  \
                        using alps::numeric::operator+;                             \
                        using alps::numeric::operator-;                             \
                        using alps::numeric::operator*;                             \
                        using alps::numeric::operator/;

                    #define NUMERIC_FUNCTION_IMPLEMENTATION(FUNCTION_NAME, ERROR)    \
                        void FUNCTION_NAME () {                                      \
                            B:: FUNCTION_NAME ();                                    \
                            NUMERIC_FUNCTION_USING                                   \
                            m_error = ERROR ;                                        \
                        }

                    NUMERIC_FUNCTION_IMPLEMENTATION(sin, abs(cos(this->mean()) * m_error))
                    NUMERIC_FUNCTION_IMPLEMENTATION(cos, abs(-sin(this->mean()) * m_error))
                    NUMERIC_FUNCTION_IMPLEMENTATION(tan, abs(error_scalar_type(1) / (cos(this->mean()) * cos(this->mean())) * m_error))
                    NUMERIC_FUNCTION_IMPLEMENTATION(sinh, abs(cosh(this->mean()) * m_error))
                    NUMERIC_FUNCTION_IMPLEMENTATION(cosh, abs(sinh(this->mean()) * m_error))
                    NUMERIC_FUNCTION_IMPLEMENTATION(tanh, abs(error_scalar_type(1) / (cosh(this->mean()) * cosh(this->mean())) * m_error))
                    NUMERIC_FUNCTION_IMPLEMENTATION(asin, abs(error_scalar_type(1) / sqrt(error_scalar_type(1) - this->mean() * this->mean()) * m_error))
                    NUMERIC_FUNCTION_IMPLEMENTATION(acos, abs(error_scalar_type(-1) / sqrt(error_scalar_type(1) - this->mean() * this->mean()) * m_error))
                    NUMERIC_FUNCTION_IMPLEMENTATION(atan, abs(error_scalar_type(1) / (error_scalar_type(1) + this->mean() * this->mean()) * m_error))
                    // abs does not change the error, so nothing has to be done ...
                    NUMERIC_FUNCTION_IMPLEMENTATION(sq, abs(error_scalar_type(2) * this->mean() * m_error))
                    NUMERIC_FUNCTION_IMPLEMENTATION(sqrt, abs(m_error / (error_scalar_type(2) * sqrt(this->mean()))))
                    NUMERIC_FUNCTION_IMPLEMENTATION(cb, abs(error_scalar_type(3) * sq(this->mean()) * m_error))
                    NUMERIC_FUNCTION_IMPLEMENTATION(cbrt, abs(m_error / (error_scalar_type(3) * sq(pow(this->mean(), error_scalar_type(1./3.))))))
                    NUMERIC_FUNCTION_IMPLEMENTATION(exp, exp(this->mean()) * m_error)
                    NUMERIC_FUNCTION_IMPLEMENTATION(log, abs(m_error / this->mean()))

                    #undef NUMERIC_FUNCTION_IMPLEMENTATION

                private:

                    error_type m_error;

                    template<typename U> void augaddsub (U const & arg, typename boost::disable_if<boost::is_scalar<U>, int>::type = 0) {
                        using alps::numeric::operator+;
                        m_error = m_error + arg.error();
                    }
                    template<typename U> void augaddsub (U const & arg, typename boost::enable_if<boost::is_scalar<U>, int>::type = 0) {}

                    template<typename U> void augmul (U const & arg, typename boost::disable_if<boost::is_scalar<U>, int>::type = 0) {
                        using alps::numeric::operator*;
                        using alps::numeric::operator+;
                        m_error = arg.mean() * m_error + this->mean() * arg.error();
                        B::operator*=(arg);
                    }
                    template<typename U> void augmul (U const & arg, typename boost::enable_if<boost::is_scalar<U>, int>::type = 0) {
                        using alps::numeric::operator*;
                        m_error = m_error * static_cast<typename alps::element_type<error_type>::type>(arg);
                        B::operator*=(arg);
                    }

                    template<typename U> void augdiv (U const & arg, typename boost::disable_if<boost::is_scalar<U>, int>::type = 0) {
                        using alps::numeric::operator*;
                        using alps::numeric::operator/;
                        using alps::numeric::operator+;
                        m_error = m_error / arg.mean() + this->mean() * arg.error() / (arg.mean() * arg.mean());
                        B::operator/=(arg);
                    }
                    template<typename U> void augdiv (U const & arg, typename boost::enable_if<boost::is_scalar<U>, int>::type = 0) {
                        using alps::numeric::operator/;
                        m_error = m_error / static_cast<typename alps::element_type<error_type>::type>(arg);
                        B::operator/=(arg);
                    }
            };

            template<typename T, typename B> class BaseWrapper<T, error_tag, B> : public B {
                public:
                    virtual bool has_error() const = 0;
                    virtual typename error_type<B>::type error() const = 0;
            };

            template<typename T, typename B> class DerivedWrapper<T, error_tag, B> : public B {
                public:
                    DerivedWrapper(): B() {}
                    DerivedWrapper(T const & arg): B(arg) {}

                    bool has_error() const { return has_feature<T, error_tag>::type::value; }

                    typename error_type<B>::type error() const { return detail::error_impl(this->m_data); }
            };

        }
    }
}

 #endif
