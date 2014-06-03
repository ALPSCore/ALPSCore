/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <string>
#include <cassert>

namespace alps {
    namespace detail {

        template<typename params_type, typename value_type> class paramiterator
            : public boost::forward_iterator_helper<
                  paramiterator<params_type, value_type>
                , value_type
                , std::ptrdiff_t
                , value_type *
                , value_type &
            >
        {
            public:

                paramiterator(paramiterator const & arg)
                    : params(arg.params)
                    , it(arg.it)
                {}

                paramiterator(
                      params_type & p
                    , std::vector<std::string>::const_iterator i
                )
                    : params(p)
                    , it(i)
                {}

                operator paramiterator<const params_type, const value_type>() const {
                    return paramiterator<const params_type, const value_type>(params, it);
                }

                value_type & operator*() const {
                    assert(params.values.find(*it) != params.values.end());
                    return *params.values.find(*it);
                }

                void operator++() {
                    ++it;
                }

                bool operator==(paramiterator<params_type, value_type> const & arg) const {
                    return it == arg.it;
                }

            private:

                params_type & params;
                std::vector<std::string>::const_iterator it;
        };

    }
}
