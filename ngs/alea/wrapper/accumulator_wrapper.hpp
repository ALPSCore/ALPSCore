/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2011 - 2013 by Mario Koenz <mkoenz@ethz.ch>                       *
 *                              Lukas Gamper <gamperl@gmail.com>                   *
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

#ifndef ALPS_NGS_ALEA_ACCUMULATOR_WRAPPER_HPP
#define ALPS_NGS_ALEA_ACCUMULATOR_WRAPPER_HPP

#include <alps/ngs/stacktrace.hpp>
#include <alps/ngs/alea/wrapper/base_wrapper.hpp>
#include <alps/ngs/alea/wrapper/result_wrapper.hpp>
#include <alps/ngs/alea/wrapper/derived_wrapper.hpp>
#include <alps/ngs/alea/wrapper/result_type_wrapper.hpp>
 
#ifdef ALPS_HAVE_MPI
    #include <alps/ngs/boost_mpi.hpp>
#endif

#include <boost/cstdint.hpp>
#include <boost/shared_ptr.hpp>

#include <typeinfo> //used in add_value
#include <stdexcept>

namespace alps {
    namespace accumulator {
        namespace detail {

            // class that holds the base_accumulator_wrapper pointer
            class accumulator_wrapper {
                public:
                    template<typename T> 
                    accumulator_wrapper(T arg)
                        : base_(new derived_accumulator_wrapper<T>(arg))
                    {}

                    accumulator_wrapper(accumulator_wrapper const & arg)
                        : base_(arg.base_->clone()) 
                    {}
                //------------------- normal input -------------------
                    template<typename T> void operator()(T const & value) {
                        (*base_)(value); 
                    }
                    template<typename T> accumulator_wrapper & operator<<(T const & value) {
                        (*this)(value);
                        return (*this);
                    }
                //------------------- input with weight-------------------
                    template<typename T, typename W> 
                    void operator()(T const & value, W const & weight) {
                        (*base_)(value, weight);
                    }

                    template<typename T> result_type_accumulator_wrapper<T> & get() const {
                        return base_->get<T>();
                    }

                    friend std::ostream& operator<<(std::ostream &out, const accumulator_wrapper& wrapper);

                    template <typename T> T & extract() const {
                        return (dynamic_cast<derived_accumulator_wrapper<T> &>(*base_)).accum_;
                    }

                    boost::uint64_t count() const {
                        return base_->count();
                    }

                    void save(hdf5::archive & ar) const {
                        ar[""] = *base_;
                    }

                    void load(hdf5::archive & ar) {
                        ar[""] >> *base_;
                    }

                    inline void reset() {
                        base_->reset();
                    }

                    boost::shared_ptr<result_wrapper> result() const {
                        return boost::shared_ptr<result_wrapper>(new result_wrapper(base_->result()));
                    }

#ifdef ALPS_HAVE_MPI
                    inline void collective_merge(
                          boost::mpi::communicator const & comm
                        , int root
                    ) {
                        base_->collective_merge(comm, root);
                    }

                    inline void collective_merge(
                          boost::mpi::communicator const & comm
                        , int root
                    ) const {
                        base_->collective_merge(comm, root);
                    }
#endif
                private:
                    boost::shared_ptr<base_accumulator_wrapper> base_;
            };

            inline std::ostream & operator<<(std::ostream & out, const accumulator_wrapper & m) {
                m.base_->print(out);
                return out;
            }
        }

        template <typename Accum> Accum & extract(detail::accumulator_wrapper & m) {
            return m.extract<Accum>();
        }

        template <typename Accum> inline boost::uint64_t count(Accum const & arg) {
            return arg.count();
        }

        template <typename Accum> void reset(Accum & arg) {
            return arg.reset();
        }
    }
}
#endif
