/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2013 by Lukas Gamper <gamperl@gmail.ch>                           *
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

#ifndef ALPS_NGS_ALEA_RESULT_WRAPPER_HPP
#define ALPS_NGS_ALEA_RESULT_WRAPPER_HPP

#include <alps/ngs/stacktrace.hpp>
#include <alps/ngs/alea/result.hpp>
#include <alps/ngs/alea/wrapper/base_wrapper.hpp>
#include <alps/ngs/alea/wrapper/derived_wrapper.hpp>
#include <alps/ngs/alea/wrapper/result_type_wrapper.hpp>
 
#ifdef ALPS_HAVE_MPI
    #include <alps/ngs/boost_mpi.hpp>
#endif

#include <boost/cstdint.hpp>
#include <boost/shared_ptr.hpp>

#include <typeinfo> //used in add_value
#include <stdexcept>

namespace alps{
    namespace accumulator {
        namespace detail {

            // class that holds the base_result_wrapper pointer
            class result_wrapper {
                public:
                    result_wrapper(boost::shared_ptr<base_result_wrapper> const & arg)
                        : base_(arg)
                    {}

                    result_wrapper(result_wrapper const & arg)
                        : base_(arg.base_->clone())
                    {}

                    template<typename T> result_type_result_wrapper<T> & get() const {
                        return base_->get<T>();
                    }

                    friend std::ostream& operator<<(std::ostream &out, result_wrapper const & wrapper);

                    template <typename T> T & extract() const {
                        return (dynamic_cast<derived_result_wrapper<T>& >(*base_)).accum_;
                    }

                    boost::uint64_t count() const {
                        return base_->count();
                    }

                private:
                    boost::shared_ptr<base_result_wrapper> base_;
            };

            inline std::ostream & operator<<(std::ostream & out, result_wrapper const & m) {
                m.base_->print(out);
                return out;
            }
        }

        template <typename Result> Result & extract(detail::result_wrapper & m) {
            return m.extract<Result>();
        }
    }
}
#endif
