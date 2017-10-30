/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file serialize_variant.hpp
    
    @brief Header for boost::variant serialization

    @todo FIXME: Move to utilities
*/

#ifndef ALPS_PARAMS_SERIALIZE_VARIANT_HPP_d116b6a1742b418f851e0dbb87004644
#define ALPS_PARAMS_SERIALIZE_VARIANT_HPP_d116b6a1742b418f851e0dbb87004644


#include <boost/variant.hpp>
#include <boost/mpl/for_each.hpp>
#include <utility> // for std::pair
#include <stdexcept>
#include <string>
#include <boost/lexical_cast.hpp> // for error messages
#include <boost/type_index.hpp> // for error messages

namespace alps {
    namespace detail {

        struct mem_view {
            const void* buf;
            std::size_t size;
            mem_view(const void* b, std::size_t s) : buf(b), size(s) {}
        };

        /// Class to mem-serialize and mem-deserialize type T (generic version for plain types)
        // FIXME: split in 2 and allow each to be a functor??
        template <typename T>
        struct memory_serializer {
            /// provides a view to the object as mem_view
            static mem_view to_view(const T& val) { return mem_view(&val, sizeof(val)); }

            /// reconstructs the object from the mem_view
            static T from_view(const mem_view& view)
            {
                // safety check
                if (sizeof(T)!=view.size) throw std::invalid_argument("Cannot construct type "
                                                                      + boost::typeindex::type_id<T>().pretty_name()
                                                                      + " from view: expected size "
                                                                      + boost::lexical_cast<std::string>(sizeof(T))
                                                                      + " got "
                                                                      + boost::lexical_cast<std::string>(view.size));
                return *static_cast<const T*>(view.buf);
            }
        };    


        /// Class to mem-serialize and mem-deserialize a `boost::variant` over MPL type sequence MPLSEQ
        template <typename MPLSEQ>
        class variant_serializer {
          public:
            typedef typename boost::make_variant_over<MPLSEQ>::type variant_type;

            /// Mem view of a bound variant value
            struct variant_mem_view : public mem_view {
                int which;
                variant_mem_view(int w, const mem_view& view):  mem_view(view), which(w) {}
                variant_mem_view(int w, const void* b, std::size_t s): mem_view(b,s), which(w) {}
            };

          private:
            /// Visitor to obtain mem view of a bound value
            struct view_visitor : public boost::static_visitor<mem_view> {
                template <typename T>
                mem_view operator()(const T& val) const
                {
                    return memory_serializer<T>::to_view(val);
                }
            };

            /// Functor class to reconstruct object from memory view
            struct maker {
                int ncall_;
                const variant_mem_view& view_;
                variant_type& var_;

                maker(const variant_mem_view& view, variant_type& var)
                    : ncall_(0), view_(view), var_(var)
                { }

                template <typename T>
                void operator()(const T&) {
                    if (ncall_==view_.which) {
                        var_ = memory_serializer<T>::from_view(view_);
                    }
                    ++ncall_;
                }
            };

          public:
            
            /// Dump to memory a `boost::variant` over MPL type sequence MPLSEQ
            static variant_mem_view to_view(const variant_type& var)
            {
                return variant_mem_view(var.which(), boost::apply_visitor(view_visitor(), var));
            }
            
            /// Reconstruct from memory a `boost::variant` over MPL type sequence MPLSEQ
            static variant_type from_view(const variant_mem_view& view)
            {
                variant_type var;
                boost::mpl::for_each<MPLSEQ>(maker(view, var));
                return var;
            }
        };

    } // detail::
} // alps::

#endif /* ALPS_PARAMS_SERIALIZE_VARIANT_HPP_d116b6a1742b418f851e0dbb87004644 */
