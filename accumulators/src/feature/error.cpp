/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <boost/preprocessor/tuple/to_seq.hpp>
#include <boost/preprocessor/seq/for_each.hpp>

#include <alps/config.hpp>

#include <alps/accumulators/feature/error.hpp>
#include <alps/hdf5/vector.hpp>

#define ALPS_ACCUMULATOR_VALUE_TYPES_SEQ BOOST_PP_TUPLE_TO_SEQ(ALPS_ACCUMULATOR_VALUE_TYPES_SIZE, (ALPS_ACCUMULATOR_VALUE_TYPES))

namespace alps {
    namespace accumulators {
        namespace impl {

            //
            // Accumulator<T, error_tag, B>
            //

            template<typename T, typename B>
            auto Accumulator<T, error_tag, B>::error() const -> error_type const {
                using std::sqrt;
                using alps::numeric::sqrt;
                using alps::numeric::operator/;
                using alps::numeric::operator-;
                using alps::numeric::operator*;

                // TODO: make library for scalar type
                error_scalar_type cnt = B::count();
                const error_scalar_type one=1;
                if (cnt<=one) return alps::numeric::inf<error_type>(m_sum2);
                return sqrt((m_sum2 / cnt - B::mean() * B::mean()) / (cnt - one));
            }

            template<typename T, typename B>
            void Accumulator<T, error_tag, B>::operator()(T const & val) {
                using alps::numeric::operator*;
                using alps::numeric::operator+=;
                using alps::numeric::check_size;

                B::operator()(val);
                check_size(m_sum2, val);
                m_sum2 += val * val;
            }

            template<typename T, typename B>
            void Accumulator<T, error_tag, B>::save(hdf5::archive & ar) const {
                B::save(ar);
                ar["mean/error"] = error();
            }

            template<typename T, typename B>
            void Accumulator<T, error_tag, B>::load(hdf5::archive & ar) { // TODO: make archive const
                using alps::numeric::operator*;
                using alps::numeric::operator+;

                B::load(ar);
                error_type error;
                ar["mean/error"] >> error;
                // TODO: make library for scalar type
                error_scalar_type cnt = B::count();
                m_sum2 = (error * error * (cnt - static_cast<error_scalar_type>(1)) + B::mean() * B::mean()) * cnt;
            }

            template<typename T, typename B>
            bool Accumulator<T, error_tag, B>::can_load(hdf5::archive & ar) { // TODO: make archive const
                using alps::hdf5::get_extent;
                const char name[]="mean/error";
                const std::size_t ndim=std::is_scalar<T>::value? 0 : get_extent(T()).size();
                return B::can_load(ar) &&
                        detail::archive_trait<error_type>::can_load(ar, name, ndim);
            }

#ifdef ALPS_HAVE_MPI
            template<typename T, typename B>
            void Accumulator<T, error_tag, B>::collective_merge(
                  alps::mpi::communicator const & comm
                , int root
            ) {
                if (comm.rank() == root) {
                    B::collective_merge(comm, root);
                    B::reduce_if(comm, T(m_sum2), m_sum2, std::plus<typename alps::hdf5::scalar_type<T>::type>(), root);
                } else
                    const_cast<Accumulator<T, error_tag, B> const *>(this)->collective_merge(comm, root);
            }

            template<typename T, typename B>
            void Accumulator<T, error_tag, B>::collective_merge(
                  alps::mpi::communicator const & comm
                , int root
            ) const {
                B::collective_merge(comm, root);
                if (comm.rank() == root)
                    throw std::runtime_error("A const object cannot be root" + ALPS_STACKTRACE);
                else
                    B::reduce_if(comm, m_sum2, std::plus<typename alps::hdf5::scalar_type<T>::type>(), root);
            }
#endif

            #define ALPS_ACCUMULATOR_INST_ERROR_ACC(r, data, T)                                    \
                template class Accumulator<T, error_tag,                                           \
                                           Accumulator<T, mean_tag,                                \
                                           Accumulator<T, count_tag,                               \
                                           AccumulatorBase<T>>>>;
            BOOST_PP_SEQ_FOR_EACH(ALPS_ACCUMULATOR_INST_ERROR_ACC, ~, ALPS_ACCUMULATOR_VALUE_TYPES_SEQ)

            //
            // Result<T, error_tag, B>
            //

            template<typename T, typename B>
            void Result<T, error_tag, B>::save(hdf5::archive & ar) const {
                B::save(ar);
                ar["mean/error"] = error();
            }

            template<typename T, typename B>
            void Result<T, error_tag, B>::load(hdf5::archive & ar) {
                B::load(ar);
                ar["mean/error"] >> m_error;
            }

            template<typename T, typename B>
            bool Result<T, error_tag, B>::can_load(hdf5::archive & ar) { // TODO: make archive const
                using alps::hdf5::get_extent;
                const char name[]="mean/error";
                const std::size_t ndim=std::is_scalar<T>::value? 0 : get_extent(T()).size();
                return B::can_load(ar) &&
                        detail::archive_trait<error_type>::can_load(ar, name, ndim);
            }

            template<typename T, typename B>
            void Result<T, error_tag, B>::negate() {
                B::negate();
            }

            template<typename T, typename B>
            void Result<T, error_tag, B>::inverse() {
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
                template<typename T, typename B>                             \
                void Result<T, error_tag, B>:: FUNCTION_NAME () {            \
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
            NUMERIC_FUNCTION_IMPLEMENTATION(asin, abs(error_scalar_type(1) / sqrt(- this->mean() * this->mean() + error_scalar_type(1)) * m_error))
            NUMERIC_FUNCTION_IMPLEMENTATION(acos, abs(error_scalar_type(-1) / sqrt(-this->mean() * this->mean() + error_scalar_type(1) ) * m_error))
            NUMERIC_FUNCTION_IMPLEMENTATION(atan, abs(error_scalar_type(1) / (this->mean() * this->mean() +error_scalar_type(1) ) * m_error))
            // abs does not change the error, so nothing has to be done ...
            NUMERIC_FUNCTION_IMPLEMENTATION(sq, abs(this->mean() * m_error * error_scalar_type(2)))
            NUMERIC_FUNCTION_IMPLEMENTATION(sqrt, abs(m_error / (sqrt(this->mean()) * error_scalar_type(2) )))
            NUMERIC_FUNCTION_IMPLEMENTATION(cb, abs( sq(this->mean()) * m_error * error_scalar_type(3) ))
            NUMERIC_FUNCTION_IMPLEMENTATION(cbrt, abs(m_error / ( sq( cbrt(this->mean()) )*error_scalar_type(3) )))
            NUMERIC_FUNCTION_IMPLEMENTATION(exp, exp(this->mean()) * m_error)
            NUMERIC_FUNCTION_IMPLEMENTATION(log, abs(m_error / this->mean()))

            #undef NUMERIC_FUNCTION_IMPLEMENTATION

            #define ALPS_ACCUMULATOR_INST_ERROR_RESULT(r, data, T)                       \
                template class Result<T, error_tag,                                      \
                                      Result<T, mean_tag,                                \
                                      Result<T, count_tag,                               \
                                      ResultBase<T>>>>;
            BOOST_PP_SEQ_FOR_EACH(ALPS_ACCUMULATOR_INST_ERROR_RESULT, ~, ALPS_ACCUMULATOR_VALUE_TYPES_SEQ)
        }
    }
}
