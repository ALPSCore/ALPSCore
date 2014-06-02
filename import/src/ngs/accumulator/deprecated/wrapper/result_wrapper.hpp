/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

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
