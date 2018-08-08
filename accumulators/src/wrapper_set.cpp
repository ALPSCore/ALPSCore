/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/accumulators/accumulator.hpp>
#include <alps/accumulators.hpp>

namespace alps {
    namespace accumulators {

        namespace detail {
            void register_predefined_serializable_types();
        }

        namespace impl {

            template<typename T>
            wrapper_set<T>::wrapper_set() {
                std::lock_guard<std::mutex> guard(m_types_mutex);
                if (m_types.empty()) {
                    detail::register_predefined_serializable_types();
                }
            }

            template<typename T>
            T & wrapper_set<T>::operator[](std::string const & name) {
                if (!has(name))
                    m_storage.insert(make_pair(name, std::shared_ptr<T>(new T())));
                return *(m_storage.find(name)->second);
            }

            template<typename T>
            T const & wrapper_set<T>::operator[](std::string const & name) const {
                if (!has(name))
                    throw std::out_of_range("No observable found with the name: " + name + ALPS_STACKTRACE);
                return *(m_storage.find(name)->second);
            }

            template<typename T>
            bool wrapper_set<T>::has(std::string const & name) const{
                return m_storage.find(name) != m_storage.end();
            }

            template<typename T>
            void wrapper_set<T>::insert(std::string const & name, std::shared_ptr<T> ptr){
                if (has(name))
                    throw std::out_of_range("There already exists an accumulator with the name: " + name + ALPS_STACKTRACE);
                m_storage.insert(make_pair(name, ptr));
            }

            template<typename T>
            void wrapper_set<T>::print(std::ostream & os) const {
                for(const_iterator it = begin(); it != end(); ++it)
                    os << it->first << ": " << *(it->second) << std::endl;
            }

            // Explicit instantiations
            template class wrapper_set<accumulator_wrapper>;
            template class wrapper_set<result_wrapper>;
        }
    }
}
