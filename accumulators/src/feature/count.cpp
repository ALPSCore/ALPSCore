/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <boost/preprocessor/tuple/to_seq.hpp>
#include <boost/preprocessor/seq/for_each.hpp>

#include <alps/config.hpp>

#include <alps/accumulators/feature/count.hpp>

#define ALPS_ACCUMULATOR_VALUE_TYPES_SEQ BOOST_PP_TUPLE_TO_SEQ(ALPS_ACCUMULATOR_VALUE_TYPES_SIZE, (ALPS_ACCUMULATOR_VALUE_TYPES))

namespace alps {
    namespace accumulators {
        namespace impl {

            //
            // Result<T, count_tag, B>
            //

            template<typename T, typename B>
            void Result<T, count_tag, B>::operator()(T const &) {
                throw std::runtime_error("No values can be added to a result" + ALPS_STACKTRACE);
            }

            template<typename T, typename B>
            void Result<T, count_tag, B>::save(hdf5::archive & ar) const {
                if (m_count==0) {
                    throw std::logic_error("Attempt to save an empty result" + ALPS_STACKTRACE);
                }
                ar["count"] = m_count;
            }

            template<typename T, typename B>
            void Result<T, count_tag, B>::load(hdf5::archive & ar) {
                count_type cnt;
                ar["count"] >> cnt;
                if (cnt==0) {
                    throw std::runtime_error("Malformed archive containing an empty result"
                                              + ALPS_STACKTRACE);
                }
                m_count=cnt;
            }

            template<typename T, typename B>
            bool Result<T, count_tag, B>::can_load(hdf5::archive & ar) { // TODO: make archive const
                return ar.is_data("count");
            }

            template<typename T, typename B>
            void Result<T, count_tag, B>::reset() {
                throw std::runtime_error("A result cannot be reset" + ALPS_STACKTRACE);
            }

            #define ALPS_ACCUMULATOR_INST_COUNT_RESULT(r, data, T) \
                template class Result<T, count_tag, ResultBase<T>>;
            BOOST_PP_SEQ_FOR_EACH(ALPS_ACCUMULATOR_INST_COUNT_RESULT, ~, ALPS_ACCUMULATOR_VALUE_TYPES_SEQ)

            //
            // Accumulator<T, count_tag, B>
            //

            template<typename T, typename B>
            void Accumulator<T, count_tag, B>::save(hdf5::archive & ar) const {
                if (m_count==0) {
                    throw std::logic_error("Attempt to save an empty accumulator" + ALPS_STACKTRACE);
                }
                ar["count"] = m_count;
            }

            template<typename T, typename B>
            void Accumulator<T, count_tag, B>::load(hdf5::archive & ar) { // TODO: make archive const
                count_type cnt;
                ar["count"] >> cnt;
                if (cnt==0) {
                    throw std::runtime_error("Malformed archive containing an empty accumulator"
                                              + ALPS_STACKTRACE);
                }
                m_count=cnt;
            }

            template<typename T, typename B>
            bool Accumulator<T, count_tag, B>::can_load(const hdf5::archive & ar) {
                return ar.is_data("count");
            }

#ifdef ALPS_HAVE_MPI
            template<typename T, typename B>
            void Accumulator<T, count_tag, B>::collective_merge(
                  alps::mpi::communicator const & comm
                , int root
            ) {
                if (comm.rank() == root)
                    alps::alps_mpi::reduce(comm, m_count, m_count, std::plus<count_type>(), root);
                else
                    const_cast<Accumulator<T, count_tag, B> const *>(this)->collective_merge(comm, root);
            }

            template<typename T, typename B>
            void Accumulator<T, count_tag, B>::collective_merge(
                  alps::mpi::communicator const & comm
                , int root
            ) const {
                if (comm.rank() == root)
                    throw std::runtime_error("A const object cannot be root" + ALPS_STACKTRACE);
                else
                    alps::alps_mpi::reduce(comm, m_count, std::plus<count_type>(), root);
            }
#endif

            #define ALPS_ACCUMULATOR_INST_COUNT_ACC(r, data, T) \
                template class Accumulator<T, count_tag, AccumulatorBase<T>>;
            BOOST_PP_SEQ_FOR_EACH(ALPS_ACCUMULATOR_INST_COUNT_ACC, ~, ALPS_ACCUMULATOR_VALUE_TYPES_SEQ)
        }
    }
}
