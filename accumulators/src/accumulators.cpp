/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/accumulators_.hpp>
#include <alps/accumulators/accumulator.hpp>
#include <alps/accumulators/namedaccumulators.hpp>


namespace alps {
    namespace accumulators {
        namespace wrapped {


            // default constructor
            template<> virtual_result_wrapper<virtual_accumulator_wrapper>::virtual_result_wrapper()
                : m_cnt(new std::ptrdiff_t(1))
                , m_ptr(new result_wrapper())
            {}

            // constructor from raw accumulator
            template<> virtual_result_wrapper<virtual_accumulator_wrapper>::virtual_result_wrapper(result_wrapper * arg)
                : m_cnt(new std::ptrdiff_t(1))
                , m_ptr(arg)
            {}

            // copy constructor
            template<> virtual_result_wrapper<virtual_accumulator_wrapper>::virtual_result_wrapper(virtual_result_wrapper const & rhs)
                : m_cnt(rhs.m_cnt)
                , m_ptr(rhs.m_ptr)
            {
                ++(*m_cnt);
            }

            // constructor from hdf5
            template<> virtual_result_wrapper<virtual_accumulator_wrapper>::virtual_result_wrapper(hdf5::archive & ar)
                : m_cnt(new std::ptrdiff_t(1))
                , m_ptr(new result_wrapper(ar))
            {}

            // destructor
            template<> virtual_result_wrapper<virtual_accumulator_wrapper>::~virtual_result_wrapper() {
                if (!--(*m_cnt)) {
                    delete m_cnt;
                    delete m_ptr;
                }
            }

            // count
            template<> boost::uint64_t virtual_result_wrapper<virtual_accumulator_wrapper>::count() const{
                return m_ptr->count();
            }

            // mean
            #define ALPS_ACCUMULATOR_MEAN_IMPL(r, data, T)                                                      \
                template<> T virtual_result_wrapper<virtual_accumulator_wrapper>::mean_impl(T) const {          \
                    return m_ptr->mean<T>();                                                                    \
                }
            BOOST_PP_SEQ_FOR_EACH(ALPS_ACCUMULATOR_MEAN_IMPL, ~, ALPS_ACCUMULATOR_VALUE_TYPES_SEQ)
            #undef ALPS_ACCUMULATOR_MEAN_IMPL

            // error
            #define ALPS_ACCUMULATOR_ERROR_IMPL(r, data, T)                                                     \
                template<> T virtual_result_wrapper<virtual_accumulator_wrapper>::error_impl(T) const {         \
                    return m_ptr->error<T>();                                                                   \
                }
            BOOST_PP_SEQ_FOR_EACH(ALPS_ACCUMULATOR_ERROR_IMPL, ~, ALPS_ACCUMULATOR_VALUE_TYPES_SEQ)
            #undef ALPS_ACCUMULATOR_ERROR_IMPL

            // save
            template<> void virtual_result_wrapper<virtual_accumulator_wrapper>::save(hdf5::archive & ar) const {
                m_ptr->save(ar);
            }

            // load
            template<> void virtual_result_wrapper<virtual_accumulator_wrapper>::load(hdf5::archive & ar){
                m_ptr->load(ar);
            }

            // print
            template<> void virtual_result_wrapper<virtual_accumulator_wrapper>::print(std::ostream & os) const {
                m_ptr->print(os);
            }

            // default constructor
			virtual_accumulator_wrapper::virtual_accumulator_wrapper()
				: m_cnt(new std::ptrdiff_t(1))
				, m_ptr(new accumulator_wrapper())
			{}

            // constructor from raw accumulator
            virtual_accumulator_wrapper::virtual_accumulator_wrapper(accumulator_wrapper * arg)
				: m_cnt(new std::ptrdiff_t(1))
				, m_ptr(arg)
            {}

            // copy constructor
			virtual_accumulator_wrapper::virtual_accumulator_wrapper(virtual_accumulator_wrapper const & rhs)
				: m_cnt(rhs.m_cnt)
				, m_ptr(rhs.m_ptr)
			{
				++(*m_cnt);
			}

            // constructor from hdf5
			virtual_accumulator_wrapper::virtual_accumulator_wrapper(hdf5::archive & ar)
				: m_cnt(new std::ptrdiff_t(1))
				, m_ptr(new accumulator_wrapper(ar))
            {}

			// destructor
			virtual_accumulator_wrapper::~virtual_accumulator_wrapper() {
				if (!--(*m_cnt)) {
					delete m_cnt;
					delete m_ptr;
				}
            }

            // operator()
            #define ALPS_ACCUMULATOR_OPERATOR_CALL(r, data, T)                                              \
                virtual_accumulator_wrapper & virtual_accumulator_wrapper::operator()(T const & value) {    \
                    (*m_ptr)(value);                                                                        \
                    return (*this);                                                                         \
                }
            BOOST_PP_SEQ_FOR_EACH(ALPS_ACCUMULATOR_OPERATOR_CALL, ~, ALPS_ACCUMULATOR_VALUE_TYPES_SEQ)
            #undef ALPS_ACCUMULATOR_OPERATOR_CALL

            /// Merge another accumulator into this one. @param rhs  accumulator to merge.
			void virtual_accumulator_wrapper::merge(const virtual_accumulator_wrapper & rhs){
            	m_ptr->merge(*(rhs.m_ptr));
            }

            virtual_accumulator_wrapper & virtual_accumulator_wrapper::operator=(boost::shared_ptr<virtual_accumulator_wrapper> const & rhs){
            	(*m_ptr) = *(rhs->m_ptr);
            	return *this;
            }

            // count
            boost::uint64_t virtual_accumulator_wrapper::count() const{
            	return m_ptr->count();
            }

            // mean
            #define ALPS_ACCUMULATOR_MEAN_IMPL(r, data, T)                  \
                T virtual_accumulator_wrapper::mean_impl(T) const {         \
                    return m_ptr->mean<T>();                                \
                }
            BOOST_PP_SEQ_FOR_EACH(ALPS_ACCUMULATOR_MEAN_IMPL, ~, ALPS_ACCUMULATOR_VALUE_TYPES_SEQ)
            #undef ALPS_ACCUMULATOR_MEAN_IMPL

            // error
            #define ALPS_ACCUMULATOR_ERROR_IMPL(r, data, T)                 \
                T virtual_accumulator_wrapper::error_impl(T) const {        \
                    return m_ptr->error<T>();                               \
                }
            BOOST_PP_SEQ_FOR_EACH(ALPS_ACCUMULATOR_ERROR_IMPL, ~, ALPS_ACCUMULATOR_VALUE_TYPES_SEQ)
            #undef ALPS_ACCUMULATOR_ERROR_IMPL

            // save
            void virtual_accumulator_wrapper::save(hdf5::archive & ar) const {
            	m_ptr->save(ar);
            }

            // load
            void virtual_accumulator_wrapper::load(hdf5::archive & ar){
            	m_ptr->load(ar);
            }

            // reset
            void virtual_accumulator_wrapper::reset() const {
            	m_ptr->reset();
            }

            // result
            boost::shared_ptr<virtual_result_wrapper<virtual_accumulator_wrapper> > virtual_accumulator_wrapper::result() const {
                return boost::shared_ptr<virtual_result_wrapper<virtual_accumulator_wrapper> >(
                    new virtual_result_wrapper<virtual_accumulator_wrapper>(new result_wrapper(*(m_ptr->result())))
                );
            }

            // print
            void virtual_accumulator_wrapper::print(std::ostream & os) const {
            	m_ptr->print(os);
            }

#ifdef ALPS_HAVE_MPI
            // collective_merge
            void virtual_accumulator_wrapper::collective_merge(alps::mpi::communicator const & comm, int root) {
            	m_ptr->collective_merge(comm, root);
            }
            void virtual_accumulator_wrapper::collective_merge(alps::mpi::communicator const & comm, int root) const {
            	m_ptr->collective_merge(comm, root);
            }
#endif
		}
	}

    #define ALPS_ACCUMULATOR_ADD_ACCUMULATOR(r, type, T)                                                                \
        accumulator_set & operator<<(accumulator_set & set, const MeanAccumulator< T > & arg) {                         \
            set.insert(arg.name(), boost::shared_ptr<accumulators::wrapped::virtual_accumulator_wrapper>(               \
                new accumulators::wrapped::virtual_accumulator_wrapper(new accumulators::accumulator_wrapper(           \
                    accumulators::impl::Accumulator<                                                                    \
                        T , accumulators::mean_tag, accumulators::impl::Accumulator<                                    \
                            T , accumulators::count_tag, accumulators::impl::AccumulatorBase< T >                       \
                        >                                                                                               \
                    >()                                                                                                 \
                )                                                                                                       \
            )));                                                                                                        \
            return set;                                                                                                 \
        }                                                                                                               \
        accumulator_set & operator<<(accumulator_set & set, const NoBinningAccumulator< T > & arg) {                    \
            set.insert(arg.name(), boost::shared_ptr<accumulators::wrapped::virtual_accumulator_wrapper>(               \
                new accumulators::wrapped::virtual_accumulator_wrapper(new accumulators::accumulator_wrapper(           \
                    accumulators::impl::Accumulator<                                                                    \
                        T , accumulators::error_tag, accumulators::impl::Accumulator<                                   \
                            T , accumulators::mean_tag, accumulators::impl::Accumulator<                                \
                                T , accumulators::count_tag, accumulators::impl::AccumulatorBase< T >                   \
                            >                                                                                           \
                        >                                                                                               \
                    >()                                                                                                 \
                )                                                                                                       \
            )));                                                                                                        \
            return set;                                                                                                 \
        }                                                                                                               \
        accumulator_set & operator<<(accumulator_set & set, const LogBinningAccumulator< T > & arg) {                    \
            set.insert(arg.name(), boost::shared_ptr<accumulators::wrapped::virtual_accumulator_wrapper>(               \
                new accumulators::wrapped::virtual_accumulator_wrapper(new accumulators::accumulator_wrapper(           \
                    accumulators::impl::Accumulator<                                                                    \
                        T , accumulators::binning_analysis_tag, accumulators::impl::Accumulator<                        \
                            T , accumulators::error_tag, accumulators::impl::Accumulator<                               \
                                T , accumulators::mean_tag, accumulators::impl::Accumulator<                            \
                                    T , accumulators::count_tag, accumulators::impl::AccumulatorBase< T >               \
                                >                                                                                       \
                            >                                                                                           \
                        >                                                                                               \
                    >()                                                                                                 \
                )                                                                                                       \
            )));                                                                                                        \
            return set;                                                                                                 \
        }                                                                                                               \
        accumulator_set & operator<<(accumulator_set & set, const FullBinningAccumulator< T > & arg) {                    \
            set.insert(arg.name(), boost::shared_ptr<accumulators::wrapped::virtual_accumulator_wrapper>(               \
                new accumulators::wrapped::virtual_accumulator_wrapper(new accumulators::accumulator_wrapper(           \
                    accumulators::impl::Accumulator<                                                                    \
                        T , accumulators::max_num_binning_tag, accumulators::impl::Accumulator<                         \
                            T , accumulators::binning_analysis_tag, accumulators::impl::Accumulator<                    \
                                T , accumulators::error_tag, accumulators::impl::Accumulator<                           \
                                    T , accumulators::mean_tag, accumulators::impl::Accumulator<                        \
                                        T , accumulators::count_tag, accumulators::impl::AccumulatorBase< T >           \
                                    >                                                                                   \
                                >                                                                                       \
                            >                                                                                           \
                        >                                                                                               \
                    >()                                                                                                 \
                )                                                                                                       \
            )));                                                                                                        \
            return set;                                                                                                 \
        }
    BOOST_PP_SEQ_FOR_EACH(ALPS_ACCUMULATOR_ADD_ACCUMULATOR, ~, ALPS_ACCUMULATOR_VALUE_TYPES_SEQ)
    #undef ALPS_ACCUMULATOR_ADD_ACCUMULATOR
}
