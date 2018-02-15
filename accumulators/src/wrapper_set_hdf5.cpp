/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <boost/preprocessor/tuple/to_seq.hpp>
#include <boost/preprocessor/seq/for_each.hpp>

#include <alps/config.hpp>

#include <alps/accumulators/wrapper_set.hpp>
#include <alps/accumulators.hpp>

#include <alps/hdf5/vector.hpp>

#define ALPS_ACCUMULATOR_VALUE_TYPES_SEQ BOOST_PP_TUPLE_TO_SEQ(ALPS_ACCUMULATOR_VALUE_TYPES_SIZE, (ALPS_ACCUMULATOR_VALUE_TYPES))

namespace alps {
    namespace accumulators {
         namespace detail {

            template<typename T> struct serializable_type {
                virtual ~serializable_type() {}
                virtual std::size_t rank() const = 0;
                virtual bool can_load(hdf5::archive & ar) const = 0;
                virtual T * create(hdf5::archive & ar) const = 0;
            };

            template<typename T, typename A> struct serializable_type_impl : public serializable_type<T> {
                std::size_t rank() const {
                    return A::rank();
                }
                bool can_load(hdf5::archive & ar) const {
                    return A::can_load(ar);
                }
                T * create(hdf5::archive & /*ar*/) const {
                    return new T(A());
                }
            };

            void register_predefined_serializable_types();
        }

        namespace impl {
            /// Register a serializable type, without locking
            template<typename T> template<typename A> void wrapper_set<T>::register_serializable_type_nolock() {
                m_types.push_back(boost::shared_ptr<detail::serializable_type<T> >(new detail::serializable_type_impl<T, A>));
                for (std::size_t i = m_types.size(); i > 1 && m_types[i - 1]->rank() > m_types[i - 2]->rank(); --i)
                    m_types[i - 1].swap(m_types[i - 2]);
            }

            /// Register a user-defined serializable type
            template<typename T> template<typename A> void wrapper_set<T>::register_serializable_type() {
                std::lock_guard<std::mutex> guard(m_types_mutex);
                if (m_types.empty()) detail::register_predefined_serializable_types();
                register_serializable_type_nolock<A>();
            }
        }

        namespace detail {
            void register_predefined_serializable_types() {
                #define ALPS_ACCUMULATOR_REGISTER_ACCUMULATOR(A)                                                    \
                    accumulator_set::register_serializable_type_nolock<A::accumulator_type>();                      \
                    result_set::register_serializable_type_nolock<A::result_type>();

                #define ALPS_ACCUMULATOR_REGISTER_TYPE(r, data, T)                                                  \
                    ALPS_ACCUMULATOR_REGISTER_ACCUMULATOR(MeanAccumulator<T>)                                       \
                    ALPS_ACCUMULATOR_REGISTER_ACCUMULATOR(NoBinningAccumulator<T>)                                  \
                    ALPS_ACCUMULATOR_REGISTER_ACCUMULATOR(LogBinningAccumulator<T>)                                 \
                    ALPS_ACCUMULATOR_REGISTER_ACCUMULATOR(FullBinningAccumulator<T>)

                BOOST_PP_SEQ_FOR_EACH(ALPS_ACCUMULATOR_REGISTER_TYPE, ~, ALPS_ACCUMULATOR_VALUE_TYPES_SEQ)

                #undef ALPS_ACCUMULATOR_REGISTER_TYPE
                #undef ALPS_ACCUMULATOR_REGISTER_ACCUMULATOR
            }
        }

        namespace impl {

            template<typename T>
            void wrapper_set<T>::save(hdf5::archive & ar) const {
                ar.create_group("");
                for(const_iterator it = begin(); it != end(); ++it) {
                    if (it->second->count()!=0) {
                        ar[it->first] = *(it->second);
                    }
                }
            }

            template<typename T>
            void wrapper_set<T>::load(hdf5::archive & ar) {
                std::lock_guard<std::mutex> guard(m_types_mutex);
                std::vector<std::string> list = ar.list_children("");
                for (std::vector<std::string>::const_iterator it = list.begin(); it != list.end(); ++it) {
                    ar.set_context(*it);
                    for (typename std::vector<boost::shared_ptr<detail::serializable_type<T> > >::const_iterator jt = m_types.begin()
                        ; jt != m_types.end()
                        ; ++jt
                    )
                        if ((*jt)->can_load(ar)) {
                            operator[](*it) = boost::shared_ptr<T>((*jt)->create(ar));
                            break;
                        }
                    if (!has(*it))
                        throw std::logic_error("The Accumulator/Result " + *it + " cannot be unserilized" + ALPS_STACKTRACE);
                    operator[](*it).load(ar);
                    ar.set_context("..");
                }
            }

            template void wrapper_set<accumulator_wrapper>::save(hdf5::archive &) const;
            template void wrapper_set<result_wrapper>::save(hdf5::archive &) const;
            template void wrapper_set<accumulator_wrapper>::load(hdf5::archive &);
            template void wrapper_set<result_wrapper>::load(hdf5::archive &);
        }
    }
}
