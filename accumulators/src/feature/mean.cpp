/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <boost/preprocessor/tuple/to_seq.hpp>
#include <boost/preprocessor/seq/for_each.hpp>

#include <alps/config.hpp>

#include <alps/accumulators/feature/mean.hpp>
#include <alps/hdf5/vector.hpp>

#define ALPS_ACCUMULATOR_VALUE_TYPES_SEQ BOOST_PP_TUPLE_TO_SEQ(ALPS_ACCUMULATOR_VALUE_TYPES_SIZE, (ALPS_ACCUMULATOR_VALUE_TYPES))

namespace alps {
    namespace accumulators {
        namespace impl {

            //
            // Accumulator<T, mean_tag, B>
            //

            template<typename T, typename B>
            auto Accumulator<T, mean_tag, B>::mean() const -> mean_type const {
                using alps::numeric::operator/;

                // TODO: make library for scalar type
                typename alps::numeric::scalar<mean_type>::type cnt = B::count();

                return mean_type(m_sum) / cnt;
            }

            template<typename T, typename B>
            void Accumulator<T, mean_tag, B>::operator()(T const & val) {
                using alps::numeric::operator+=;
                using alps::numeric::check_size;

                B::operator()(val);
                check_size(m_sum, val);
                m_sum += val;
            }

            template<typename T, typename B>
            void Accumulator<T, mean_tag, B>::save(hdf5::archive & ar) const {
                B::save(ar);
                ar["mean/value"] = mean();
            }

            template<typename T, typename B>
            void Accumulator<T, mean_tag, B>::load(hdf5::archive & ar) { // TODO: make archive const
                using alps::numeric::operator*;

                B::load(ar);
                mean_type mean;
                ar["mean/value"] >> mean;
                // TODO: make library for scalar type
                typename alps::numeric::scalar<mean_type>::type cnt = B::count();
                m_sum = mean * cnt;
            }

            template<typename T, typename B>
            bool Accumulator<T, mean_tag, B>::can_load(hdf5::archive & ar) { // TODO: make archive const
                using alps::hdf5::get_extent;
                const char name[]="mean/value";
                const std::size_t ndim=std::is_scalar<T>::value? 0 : get_extent(T()).size();
                return B::can_load(ar) &&
                        detail::archive_trait<mean_type>::can_load(ar, name, ndim);
            }


#ifdef ALPS_HAVE_MPI
            template<typename T, typename B>
            void Accumulator<T, mean_tag, B>::collective_merge(
                  alps::mpi::communicator const & comm
                , int root
            ) {
                if (comm.rank() == root) {
                    B::collective_merge(comm, root);
                    B::reduce_if(comm, T(m_sum), m_sum, std::plus<typename alps::hdf5::scalar_type<T>::type>(), root);
                } else
                    const_cast<Accumulator<T, mean_tag, B> const *>(this)->collective_merge(comm, root);
            }

            template<typename T, typename B>
            void Accumulator<T, mean_tag, B>::collective_merge(
                  alps::mpi::communicator const & comm
                , int root
            ) const {
                B::collective_merge(comm, root);
                if (comm.rank() == root)
                    throw std::runtime_error("A const object cannot be root" + ALPS_STACKTRACE);
                else
                    B::reduce_if(comm, m_sum, std::plus<typename alps::hdf5::scalar_type<T>::type>(), root);
            }
#endif

            template<typename T, typename B>
            T const & Accumulator<T, mean_tag, B>::sum() const {
                return m_sum;
            }

            #define ALPS_ACCUMULATOR_INST_MEAN_ACC(r, data, T)                                    \
                template class Accumulator<T, mean_tag,                                           \
                                           Accumulator<T, count_tag, AccumulatorBase<T>>>;
            BOOST_PP_SEQ_FOR_EACH(ALPS_ACCUMULATOR_INST_MEAN_ACC, ~, ALPS_ACCUMULATOR_VALUE_TYPES_SEQ)

            //
            // Result<T, mean_tag, B>
            //

            template<typename T, typename B>
            void Result<T, mean_tag, B>::save(hdf5::archive & ar) const {
                B::save(ar);
                ar["mean/value"] = mean();
            }

            template<typename T, typename B>
            void Result<T, mean_tag, B>::load(hdf5::archive & ar) {
                B::load(ar);
                ar["mean/value"] >> m_mean;
            }

            template<typename T, typename B>
            bool Result<T, mean_tag, B>::can_load(hdf5::archive & ar) { // TODO: make archive const
                using alps::hdf5::get_extent;
                const char name[]="mean/value";
                const std::size_t ndim=std::is_scalar<T>::value? 0 : get_extent(T()).size();
                return B::can_load(ar) &&
                        detail::archive_trait<mean_type>::can_load(ar, name, ndim);
            }

            template<typename T, typename B>
            void Result<T, mean_tag, B>::negate() {
                using alps::numeric::operator-;
                m_mean = -m_mean;
                B::negate();
            }

            template<typename T, typename B>
            void Result<T, mean_tag, B>::inverse() {
                using alps::numeric::operator/;
                // TODO: make library for scalar type
                typename alps::numeric::scalar<mean_type>::type one = 1;
                m_mean = one / m_mean;
                B::inverse();
            }

            #define NUMERIC_FUNCTION_IMPLEMENTATION(FUNCTION_NAME)              \
                template<typename T, typename B>                                \
                void Result<T, mean_tag, B>:: FUNCTION_NAME () {                \
                    B:: FUNCTION_NAME ();                                       \
                    using std:: FUNCTION_NAME ;                                 \
                    using alps::numeric:: FUNCTION_NAME ;                       \
                    m_mean = FUNCTION_NAME (m_mean);                            \
                }

            NUMERIC_FUNCTION_IMPLEMENTATION(sin)
            NUMERIC_FUNCTION_IMPLEMENTATION(cos)
            NUMERIC_FUNCTION_IMPLEMENTATION(tan)
            NUMERIC_FUNCTION_IMPLEMENTATION(sinh)
            NUMERIC_FUNCTION_IMPLEMENTATION(cosh)
            NUMERIC_FUNCTION_IMPLEMENTATION(tanh)
            NUMERIC_FUNCTION_IMPLEMENTATION(asin)
            NUMERIC_FUNCTION_IMPLEMENTATION(acos)
            NUMERIC_FUNCTION_IMPLEMENTATION(atan)
            NUMERIC_FUNCTION_IMPLEMENTATION(abs)
            NUMERIC_FUNCTION_IMPLEMENTATION(sqrt)
            NUMERIC_FUNCTION_IMPLEMENTATION(log)

            #undef NUMERIC_FUNCTION_IMPLEMENTATION

            #define NUMERIC_FUNCTION_IMPLEMENTATION(FUNCTION_NAME)          \
                template<typename T, typename B>                            \
                void Result<T, mean_tag, B>:: FUNCTION_NAME () {            \
                    B:: FUNCTION_NAME ();                                   \
                    using alps::numeric:: FUNCTION_NAME ;                   \
                    using alps::numeric:: FUNCTION_NAME ;                   \
                    m_mean = FUNCTION_NAME (m_mean);                        \
                }

            NUMERIC_FUNCTION_IMPLEMENTATION(sq)
            NUMERIC_FUNCTION_IMPLEMENTATION(cb)
            NUMERIC_FUNCTION_IMPLEMENTATION(cbrt)

            #undef NUMERIC_FUNCTION_IMPLEMENTATION

            #define ALPS_ACCUMULATOR_INST_MEAN_RESULT(r, data, T)                            \
                template class Result<T, mean_tag,                                           \
                                      Result<T, count_tag, ResultBase<T>>>;
            BOOST_PP_SEQ_FOR_EACH(ALPS_ACCUMULATOR_INST_MEAN_RESULT, ~, ALPS_ACCUMULATOR_VALUE_TYPES_SEQ)
        }
    }
}
