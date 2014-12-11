/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/accumulators_.hpp>
#include <alps/accumulators/accumulator.hpp>
#include <alps/accumulators/namedaccumulators.hpp>


namespace alps {
    namespace accumulators {
        namespace wrapped {

        	// TODO: make reference counter ...
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
            virtual_accumulator_wrapper &  virtual_accumulator_wrapper::operator()(double const & value) {
                (*m_ptr)(value);
                return (*this);
            }

            /// Merge another accumulator into this one. @param rhs_acc  accumulator to merge.
			void virtual_accumulator_wrapper::merge(const virtual_accumulator_wrapper & rhs){
            	m_ptr->merge(*(rhs.m_ptr));
            }

            virtual_accumulator_wrapper & virtual_accumulator_wrapper::operator=(boost::shared_ptr<virtual_accumulator_wrapper> const & rhs){
            	(*m_ptr) = *(rhs->m_ptr);
            	return *this;
            }

                // get
                    // template <typename T> base_wrapper<T> & get() {
                    //     get_visitor<T> visitor;
                    //     boost::apply_visitor(visitor, m_variant);
                    //     return *visitor.value;
                    // }

                // extract
                    // template <typename A> A & extract() {
                    //     throw std::logic_error(std::string("unknown type : ") + typeid(A).name() + ALPS_STACKTRACE);
                    // }
                    // template <> MeanAccumulatorDouble & extract<MeanAccumulatorDouble>();

            // count
            boost::uint64_t virtual_accumulator_wrapper::count() const{
            	return m_ptr->count();
            }

                // // mean, error
                // #define ALPS_ACCUMULATOR_PROPERTY_PROXY(PROPERTY, TYPE)                                                 \
                //     private:                                                                                            \
                //         template<typename T> struct PROPERTY ## _visitor: public boost::static_visitor<> {              \
                //             template<typename X> void apply(typename boost::enable_if<                                  \
                //                 typename detail::is_valid_argument<typename TYPE <X>::type, T>::type, X const &         \
                //             >::type arg) const {                                                                        \
                //                 value = arg. PROPERTY ();                                                               \
                //             }                                                                                           \
                //             template<typename X> void apply(typename boost::disable_if<                                 \
                //                 typename detail::is_valid_argument<typename TYPE <X>::type, T>::type, X const &         \
                //             >::type arg) const {                                                                        \
                //                 throw std::logic_error(std::string("cannot convert: ")                                  \
                //                     + typeid(typename TYPE <X>::type).name() + " to "                                   \
                //                     + typeid(T).name() + ALPS_STACKTRACE);                                              \
                //             }                                                                                           \
                //             template<typename X> void operator()(X const & arg) const {                                 \
                //                 apply<typename X::element_type>(*arg);                                                  \
                //             }                                                                                           \
                //             mutable T value;                                                                            \
                //         };                                                                                              \
                //     public:                                                                                             \
                //         template<typename T> typename TYPE <base_wrapper<T> >::type PROPERTY () const {                 \
                //             PROPERTY ## _visitor<typename TYPE <base_wrapper<T> >::type> visitor;                       \
                //             boost::apply_visitor(visitor, m_variant);                                                   \
                //             return visitor.value;                                                                       \
                //         }
                // ALPS_ACCUMULATOR_PROPERTY_PROXY(mean, mean_type)
                // ALPS_ACCUMULATOR_PROPERTY_PROXY(error, error_type)
                // #undef ALPS_ACCUMULATOR_FUNCTION_PROXY

            // save
            void virtual_accumulator_wrapper::save(hdf5::archive & ar) const{
            	m_ptr->save(ar);
            }

            // load
            void virtual_accumulator_wrapper::load(hdf5::archive & ar){
            	m_ptr->load(ar);
            }

            // reset
            void virtual_accumulator_wrapper::reset() const{
            	m_ptr->reset();
            }

                // result
                    // boost::shared_ptr<result_wrapper> result() const;

            // print
            void virtual_accumulator_wrapper::print(std::ostream & os) const{
            	m_ptr->print(os);
            }

#ifdef ALPS_HAVE_MPI
            // collective_merge
            void virtual_accumulator_wrapper::collective_merge(boost::mpi::communicator const & comm, int root) {
            	m_ptr->collective_merge(comm, root);
            }
            void virtual_accumulator_wrapper::collective_merge(boost::mpi::communicator const & comm, int root) const {
            	m_ptr->collective_merge(comm, root);
            }
#endif
		}
	}

    accumulator_set & operator<<(accumulator_set & set, const MeanAccumulator<double> & arg) {
        set.insert(arg.name(), boost::shared_ptr<accumulators::wrapped::virtual_accumulator_wrapper>(
        	new accumulators::wrapped::virtual_accumulator_wrapper(new accumulators::accumulator_wrapper(
				accumulators::impl::Accumulator<
					double, accumulators::mean_tag, accumulators::impl::Accumulator<
						double, accumulators::count_tag, accumulators::impl::AccumulatorBase<double> 
					> 
				>()
			)
        )));
        return set;
    }
}
