/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

//#include <alps/accumulators/wrapper_set.hpp>
#include <alps/accumulators.hpp>

#include <alps/hdf5/vector.hpp>

namespace alps {
    namespace accumulators {
        namespace detail {
            void register_predefined_serializable_types() {
                #define ALPS_ACCUMULATOR_REGISTER_ACCUMULATOR(A)                                                    \
                    accumulator_set::register_serializable_type_nolock<A::accumulator_type>();                         \
                    result_set::register_serializable_type_nolock<A::result_type>();

                #define ALPS_ACCUMULATOR_REGISTER_TYPE(T)                                                           \
                    ALPS_ACCUMULATOR_REGISTER_ACCUMULATOR(MeanAccumulator<T>)                                       \
                    ALPS_ACCUMULATOR_REGISTER_ACCUMULATOR(NoBinningAccumulator<T>)                                  \
                    ALPS_ACCUMULATOR_REGISTER_ACCUMULATOR(LogBinningAccumulator<T>)                                 \
                    ALPS_ACCUMULATOR_REGISTER_ACCUMULATOR(FullBinningAccumulator<T>)

                // TODO: use ALPS_ACCUMULATOR_VALUE_TYPES and iterate over it
                ALPS_ACCUMULATOR_REGISTER_TYPE(float)
                ALPS_ACCUMULATOR_REGISTER_TYPE(double)
                ALPS_ACCUMULATOR_REGISTER_TYPE(long double)
                ALPS_ACCUMULATOR_REGISTER_TYPE(std::vector<float>)
                ALPS_ACCUMULATOR_REGISTER_TYPE(std::vector<double>)
                ALPS_ACCUMULATOR_REGISTER_TYPE(std::vector<long double>)

                #undef ALPS_ACCUMULATOR_REGISTER_TYPE
                #undef ALPS_ACCUMULATOR_REGISTER_ACCUMULATOR
            }
        }

        namespace impl {

            /// Register a serializable type, without locking
            template<typename T> template<typename A> void wrapper_set<T>::register_serializable_type_nolock() {
                m_types.push_back(boost::shared_ptr<T>(new T(A())));
                for (std::size_t i = m_types.size(); i > 1 && m_types[i - 1]->rank() > m_types[i - 2]->rank(); --i)
                    m_types[i - 1].swap(m_types[i - 2]);
            }

            /// Register a user-defined serializable type
            template<typename T> template<typename A> void wrapper_set<T>::register_serializable_type() {
                std::lock_guard<std::mutex> guard(m_types_mutex);
                if (m_types.empty()) detail::register_predefined_serializable_types();
                register_serializable_type_nolock<A>();
            }

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

            template class wrapper_set<accumulator_wrapper>;
            //template class wrapper_set<result_wrapper>;
        }
    }
}
