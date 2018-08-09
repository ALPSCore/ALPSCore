/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
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
#include <stdexcept>
#include <boost/optional.hpp>

namespace alps {
    namespace detail {

        /// Class to mem-serialize and mem-deserialize a `boost::variant` over MPL type sequence MPLSEQ
        template <typename MPLSEQ, typename CONSUMER, typename PRODUCER>
        class variant_serializer {
          public:
            typedef typename boost::make_variant_over<MPLSEQ>::type variant_type;
            typedef CONSUMER consumer_type;
            typedef PRODUCER producer_type;

          private:
            /// Visitor to call consumer on the bound value
            struct consume_visitor : public boost::static_visitor<> {
                consumer_type& consumer_;

                consume_visitor(consumer_type& consumer) : consumer_(consumer) {}
                
                template <typename T>
                void operator()(const T& val) const
                {
                    consumer_(val);
                }
            };

            /// Functor class to reconstruct object via producer
            struct maker {
                producer_type& producer_;
                variant_type& var_;

                maker(producer_type& producer, variant_type& var)
                    : producer_(producer), var_(var)
                { }

                // Always assigns --- therefore, applies last successful outcome!
                template <typename T>
                void operator()(const T&) {
                    boost::optional<T> maybe_val=producer_((T*)0);
                    if (maybe_val) var_=*maybe_val;
                }
            };

          public:
            
            /// Send the variant to a consumer
            static void consume(consumer_type& consumer, const variant_type& var)
            {
                boost::apply_visitor(consume_visitor(consumer), var);
            }
            
            /// Receive the variant from a producer
            static variant_type produce(producer_type& producer)
            {
                variant_type var;
                boost::mpl::for_each<MPLSEQ>(maker(producer, var));
                return var;
            }
        };

    } // detail::
} // alps::

#endif /* ALPS_PARAMS_SERIALIZE_VARIANT_HPP_d116b6a1742b418f851e0dbb87004644 */
