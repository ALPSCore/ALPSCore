/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2011 - 2013 by Mario Koenz <mkoenz@ethz.ch>                       *
 *                              Lukas Gamper <gamperl@gmail.com>                   *
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

#ifndef ALPS_NGS_ACCUMULATOR_ERROR_HPP
#define ALPS_NGS_ACCUMULATOR_ERROR_HPP

#include <alps/ngs/accumulator/feature.hpp>
#include <alps/ngs/accumulator/feature/mean.hpp>
#include <alps/ngs/accumulator/feature/count.hpp>

#include <alps/hdf5/archive.hpp>
#include <alps/ngs/numeric.hpp>
#include <alps/ngs/stacktrace.hpp>
#include <alps/ngs/short_print.hpp>

#include <boost/mpl/if.hpp>
#include <boost/utility.hpp>
#include <boost/type_traits/is_scalar.hpp>
#include <boost/type_traits/is_integral.hpp>

#include <stdexcept>

namespace alps {
    namespace accumulator {
        // this should be called namespace tag { struct error; }
        // but gcc <= 4.4 has lookup error, so name it different
        struct error_tag;

        template<typename T> struct error_type : public mean_type<T> {};

        template<typename T> struct has_feature<T, error_tag> {
            template<typename R, typename C> static char helper(R(C::*)() const);
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
                    typedef typename alps::accumulator::error_type<B>::type error_type;
                    typedef Result<T, error_tag, typename B::result_type> result_type;

                    // TODO: implement using disable_if<Accumulator<...> > ...
                    // template<typename ArgumentPack> Accumulator(ArgumentPack const & args): B(args), m_sum2(T()) {}

                    Accumulator(): B(), m_sum2(T()) {}
                    Accumulator(Accumulator const & arg): B(arg), m_sum2(arg.m_sum2) {}

                    error_type const error() const {
                        using std::sqrt;
                        using alps::ngs::numeric::sqrt;
                        using alps::ngs::numeric::operator/;
                        using alps::ngs::numeric::operator-;
                        using alps::ngs::numeric::operator*;

                        // TODO: make library for scalar type
                        typename alps::hdf5::scalar_type<error_type>::type cnt = B::count();
                        return sqrt((m_sum2 / cnt - B::mean() * B::mean()) / (cnt - 1));
                    }

                    void operator()(T const & val) {
                        using alps::ngs::numeric::operator*;
                        using alps::ngs::numeric::operator+=;
                        using alps::ngs::numeric::detail::check_size;

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
                        using alps::ngs::numeric::operator*;
                        using alps::ngs::numeric::operator+;

                        B::load(ar);
                        error_type error;
                        ar["mean/error"] >> error;
                        // TODO: make library for scalar type
                        typename alps::hdf5::scalar_type<error_type>::type cnt = B::count();
                        m_sum2 = (error * error * (cnt - 1) + B::mean() * B::mean()) * cnt;
                    }

                    static std::size_t rank() { return B::rank() + 1; }
                    static bool can_load(hdf5::archive & ar) { // TODO: make archive const
                        using alps::hdf5::get_extent;

                        return B::can_load(ar)
                            && ar.is_data("mean/error") 
                            && boost::is_scalar<T>::value == ar.is_scalar("mean/error")
                            && (boost::is_scalar<T>::value || get_extent(T()).size() == ar.dimensions("mean/error"))
                        ;
                    }

                    void reset() {
                        B::reset();
                        m_sum2 = T();
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
                    typedef typename alps::accumulator::error_type<B>::type error_type;

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

                        return B::can_load(ar) 
                            && ar.is_data("mean/error") 
                            && boost::is_scalar<T>::value == ar.is_scalar("mean/error")
                            && (boost::is_scalar<T>::value || get_extent(T()).size() == ar.dimensions("mean/error"))
                        ;
                    }

                    // TODO: implement -=, *=, /=
                    // TODO: implement operators with double
                    template<typename U> void operator+=(U const & arg) { augadd(arg); }

                    #define NUMERIC_FUNCTION_USEING                                 \
                        using alps::numeric::sq;                                    \
                        using alps::ngs::numeric::cbrt;                             \
                        using alps::ngs::numeric::cb;                               \
                        using std::sqrt;                                            \
                        using alps::ngs::numeric::sqrt;                             \
                        using std::exp;                                             \
                        using alps::ngs::numeric::exp;                              \
                        using std::log;                                             \
                        using alps::ngs::numeric::log;                              \
                        using std::abs;                                             \
                        using alps::ngs::numeric::abs;                              \
                        using std::pow;                                             \
                        using alps::ngs::numeric::pow;                              \
                        using std::sin;                                             \
                        using alps::ngs::numeric::sin;                              \
                        using std::cos;                                             \
                        using alps::ngs::numeric::cos;                              \
                        using std::tan;                                             \
                        using alps::ngs::numeric::tan;                              \
                        using std::sinh;                                            \
                        using alps::ngs::numeric::sinh;                             \
                        using std::cosh;                                            \
                        using alps::ngs::numeric::cosh;                             \
                        using std::tanh;                                            \
                        using alps::ngs::numeric::tanh;                             \
                        using std::asin;                                            \
                        using alps::ngs::numeric::asin;                             \
                        using std::acos;                                            \
                        using alps::ngs::numeric::acos;                             \
                        using std::atan;                                            \
                        using alps::ngs::numeric::atan;                             \
                        using alps::ngs::numeric::operator+;                        \
                        using alps::ngs::numeric::operator-;                        \
                        using alps::ngs::numeric::operator*;                        \
                        using alps::ngs::numeric::operator/;
/*
                    #define NUMERIC_FUNCTION_OPERATOR(OP_NAME, OP_ARG)                \
                        template<typename U> void OP_NAME (U const & arg) {         \
                            B:: OP_NAME (arg);                                      \
                            OP_NAME ## _impl(arg);                                  \
                        }                                                           \
                        void OP_NAME (T const & arg) {                              \
                            B:: OP_NAME (arg);                                      \
                            NUMERIC_FUNCTION_USEING                                 \
                            m_error = OP_ARG;                                       \
                        }                                                           \

                    // addig does not change the error, so nothing has to be done ...
                    // substract does not change the error, so nothing has to be done ...
                    NUMERIC_FUNCTION_OPERATOR(muleq, sqrt(sq(arg) * sq(error())))
                    NUMERIC_FUNCTION_OPERATOR(diveq, sqrt(sq(arg) * sq(error())) / sq(arg))

                    #undef NUMERIC_FUNCTION_OPERATOR
*/
                    #define NUMERIC_FUNCTION_IMPLEMENTATION(FUNCTION_NAME, ERROR)    \
                        void FUNCTION_NAME () {                                        \
                            B:: FUNCTION_NAME ();                                    \
                            NUMERIC_FUNCTION_USEING                                    \
                            m_error = ERROR ;                                        \
                        }

                    NUMERIC_FUNCTION_IMPLEMENTATION(sin, abs(cos(this->mean()) * m_error))
                    // NUMERIC_FUNCTION_IMPLEMENTATION(cos, abs(-sin(this->mean()) * m_error)) // TODO: unary minus is missing
                    // NUMERIC_FUNCTION_IMPLEMENTATION(tan, abs(1. / (cos(this->mean()) * cos(this->mean())) * m_error)) // TODO: scalar devided by data missing
                    NUMERIC_FUNCTION_IMPLEMENTATION(sinh, abs(cosh(this->mean()) * m_error))
                    NUMERIC_FUNCTION_IMPLEMENTATION(cosh, abs(sinh(this->mean()) * m_error))
                    // NUMERIC_FUNCTION_IMPLEMENTATION(tanh, abs(1. / (cosh(this->mean()) * cosh(this->mean())) * m_error)) // TODO: scalar devided by data missing
                    // NUMERIC_FUNCTION_IMPLEMENTATION(asin, abs(1. / sqrt(1. - this->mean() * this->mean()) * m_error)) // TODO: substraction of scalar is missing
                    // NUMERIC_FUNCTION_IMPLEMENTATION(acos, abs(-1. / sqrt(1. - this->mean() * this->mean()) * m_error)) // TODO: substraction of scalar is missing
                    // NUMERIC_FUNCTION_IMPLEMENTATION(atan, abs(1. / (1. + this->mean() * this->mean()) * m_error)) // TODO: addition of scalar is missing
                    // abs does not change the error, so nothing has to be done ...
                    NUMERIC_FUNCTION_IMPLEMENTATION(sq, abs(2. * this->mean() * m_error))
                    NUMERIC_FUNCTION_IMPLEMENTATION(sqrt, abs(m_error / (2. * sqrt(this->mean()))))
                    NUMERIC_FUNCTION_IMPLEMENTATION(cb, abs(3. * sq(this->mean()) * m_error))
                    NUMERIC_FUNCTION_IMPLEMENTATION(cbrt, abs(m_error / (3. * sq(pow(this->mean(),1. / 3)))))
                    NUMERIC_FUNCTION_IMPLEMENTATION(exp, exp(this->mean()) * m_error)
                    NUMERIC_FUNCTION_IMPLEMENTATION(log, abs(m_error / this->mean()))

                    #undef NUMERIC_FUNCTION_IMPLEMENTATION

                private:

                    error_type m_error;

                    template<typename U> void augadd(U const & arg, typename boost::enable_if<boost::is_scalar<U>, int>::type = 0) {
                        B::operator+=(arg);
                    }
                    template<typename U> void augadd(U const & arg, typename boost::disable_if<boost::is_scalar<U>, int>::type = 0) {
                        using alps::numeric::sq;
                        using std::sqrt;
                        using alps::ngs::numeric::sqrt;

                        using alps::ngs::numeric::operator+;

                        m_error = sqrt(sq(m_error) + sq(arg.error()));
                        B::operator+=(arg);
                    }

                    // #define NUMERIC_FUNCTION_OPERATOR_IMPL(OP_NAME, OP_T, OP_ARG)                            \
                    //     template<typename U> void OP_NAME ## _impl(typename boost::disable_if<                \
                    //           typename boost::is_same<typename alps::hdf5::scalar_type<T>::type, U>::type     \
                    //         , U                                                                                \
                    //     >::type const & arg) {                                                                \
                    //         NUMERIC_FUNCTION_USEING                                                            \
                    //         m_error = OP_T;                                                                    \
                    //     }                                                                                    \
                    //     template<typename U> void OP_NAME ## _impl(typename boost::enable_if<                \
                    //           typename boost::is_same<typename alps::hdf5::scalar_type<T>::type, U>::type    \
                    //         , U                                                                                \
                    //     >::type arg) {                                                                        \
                    //         NUMERIC_FUNCTION_USEING                                                            \
                    //         m_error = OP_ARG;                                                                \
                    //     }

                    // NUMERIC_FUNCTION_OPERATOR_IMPL(addeq, sqrt(sq(error()) + sq(arg.error())), error())
                    // NUMERIC_FUNCTION_OPERATOR_IMPL(subeq, sqrt(sq(error()) + sq(arg.error())), error())
                    // NUMERIC_FUNCTION_OPERATOR_IMPL(muleq, sqrt(sq(arg.mean()) * sq(error()) + sq(B::mean()) * sq(arg.error())), sqrt(sq(arg) * sq(error())))
                    // NUMERIC_FUNCTION_OPERATOR_IMPL(diveq, sqrt(sq(arg.mean()) * sq(error()) + sq(B::mean()) * sq(arg.error())) / sq(arg.mean()), sqrt(sq(arg) * sq(error())) / sq(arg))

                    // #undef NUMERIC_FUNCTION_USEING
                    // #undef NUMERIC_FUNCTION_OPERATOR_IMPL
            };

            template<typename B> class BaseWrapper<error_tag, B> : public B {
                public:
                    virtual bool has_error() const = 0;
            };

            template<typename T, typename B> class ResultTypeWrapper<T, error_tag, B> : public B {
                public:
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
