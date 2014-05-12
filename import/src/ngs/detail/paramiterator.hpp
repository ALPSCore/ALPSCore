/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2010 - 2012 by Lukas Gamper <gamperl@gmail.com>                   *
 *                                                                                 *
 * This software is part of the ALPS libraries, published under the ALPS           *
 * Library License; you can use, redistribute it and/or modify it under            *
 * the terms of the license, either version 1 or (at your option) any later        *
 * version.                                                                        *
 *                                                                                 *
 * You should have received a copy of the ALPS Library License along with          *
 * the ALPS Libraries; see the file LICENSE.txt. If not, the license is also       *
 * available from http://alps.comp-phys.org/.                                      *
 *                                                                                 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,        *
 * FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT       *
 * SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE       *
 * FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,     *
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER     *
 * DEALINGS IN THE SOFTWARE.                                                       *
 *                                                                                 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

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
