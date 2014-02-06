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

// TODO: rename to type_holder

#ifndef ALPS_NGS_ALEA_DETAIL_VALUE_TYPE_IMPLEMENTATION_HEADER
#define ALPS_NGS_ALEA_DETAIL_VALUE_TYPE_IMPLEMENTATION_HEADER

#include <alps/ngs/alea/feature/feature_traits.hpp>
#include <alps/ngs/alea/feature/generate_property.hpp>

#include <boost/cstdint.hpp>

#include <typeinfo>

#ifdef ALPS_HAVE_MPI
    #include <alps/ngs/mpi.hpp>
#endif

namespace alps {
    namespace accumulator {
        //=================== value_type trait ===================
        template <typename Accum> struct value_type {
            typedef typename Accum::value_type type;
        };
        template <typename Accum> struct weight_value_type {
            typedef typename Accum::weight_value_type type;
        };
        //=================== value_type implementation ===================
        namespace detail {
            //setting up the dependencies for value_type-Implementation isn't neccessary bc has none
            template<typename T, typename W, typename base_type> class AccumulatorImplementation<type_holder<T, W>, base_type> {
                typedef AccumulatorImplementation<type_holder<T, W>, base_type> ThisType;
                public:
                    typedef T value_type;
                    typedef W weight_value_type;
                    
                    AccumulatorImplementation<type_holder<T, W>, base_type>(ThisType const & arg): count_(arg.count_) {}
                    
                    template <typename ArgumentPack>
                    AccumulatorImplementation<type_holder<T, W>, base_type>(ArgumentPack const & args, typename boost::disable_if<
                                                                                                  boost::is_base_of<ThisType, ArgumentPack>
                                                                                                , int
                                                                                                >::type = 0
                                            ): count_() 
                    {}
                    
                    inline void operator()(value_type const & val) {
                        ++count_;
                    }

                    inline ThisType& operator<<(value_type const & val) {
                        (*this)(val);
                        return (*this);
                    }
                    
                    inline boost::uint64_t const & count() const {
                        return count_; 
                    }
                
                    template<typename Stream> inline void print(Stream & os) {
                        os << "ValueType: " << typeid(value_type).name() << " " << std::endl;
                        os << "Count: " << count() << " " << std::endl;
                    }

                    void save(hdf5::archive & ar) const {
                        ar["count"] = count_;
                        ar["@valuetype"] = value_type();
                    }

                    void load(hdf5::archive & ar) {
                        ar["count"] >> count_;
                    }

                    inline void reset() {
                        count_ = 0;
                    }

#ifdef ALPS_HAVE_MPI
                    void collective_merge(
                          boost::mpi::communicator const & comm
                        , int root
                    ) {
                        if (comm.rank() == root)
                            alps::mpi::reduce(comm, count_, count_, std::plus<boost::uint64_t>(), root);
                        else
                            const_cast<ThisType const *>(this)->collective_merge(comm, root);
                    }
                    void collective_merge(
                          boost::mpi::communicator const & comm
                        , int root
                    ) const {
                        if (comm.rank() == root)
                            throw std::runtime_error("A const object cannot be root" + ALPS_STACKTRACE);

                        else
                            alps::mpi::reduce(comm, count_, std::plus<boost::uint64_t>(), root);
                    }
                protected:
                    template <typename ValueType, typename Op> void static reduce_if(
                          boost::mpi::communicator const & comm
                        , ValueType const & arg
                        , ValueType & res
                        , Op op
                        , typename boost::enable_if<typename boost::is_scalar<typename alps::hdf5::scalar_type<ValueType>::type>::type, int>::type root
                    ) {
                        alps::mpi::reduce(comm, arg, res, op, root);
                    }
                    template <typename ValueType, typename Op> void static reduce_if(
                          boost::mpi::communicator const &
                        , ValueType const &
                        , ValueType &
                        , Op
                        , typename boost::disable_if<typename boost::is_scalar<typename alps::hdf5::scalar_type<ValueType>::type>::type, int>::type
                    ) {
                        throw std::logic_error("No boost::mpi::reduce available for this type " + std::string(typeid(ValueType).name()) + ALPS_STACKTRACE);
                    }

                    template <typename ValueType, typename Op> void static reduce_if(
                          boost::mpi::communicator const & comm
                        , ValueType const & arg
                        , Op op
                        , typename boost::enable_if<typename boost::is_scalar<typename alps::hdf5::scalar_type<ValueType>::type>::type, int>::type root
                    ) {
                        alps::mpi::reduce(comm, arg, op, root);
                    }
                    template <typename ValueType, typename Op> void static reduce_if(
                          boost::mpi::communicator const &
                        , ValueType const &
                        , Op
                        , typename boost::disable_if<typename boost::is_scalar<typename alps::hdf5::scalar_type<ValueType>::type>::type, int>::type
                    ) {
                        throw std::logic_error("No boost::mpi::reduce available for this type " + std::string(typeid(ValueType).name()) + ALPS_STACKTRACE);
                    }
#endif

                private:
                    boost::uint64_t count_;
            };

            template<typename T, typename base_type> class ResultImplementation<type_holder<T>, base_type> {
                public:

                    typedef T value_type;

                    template<typename Accumulator> ResultImplementation(Accumulator const & accum)
                        : count_(accum.count())
                    {}

                    inline boost::uint64_t const & count() const {
                        return count_; 
                    }

                    template<typename Stream> inline void print(Stream & os) {
                        os << "Count: " << count() << " " << std::endl;
                    }

// TODO: implement!
                private:
                    boost::uint64_t count_;
            };

        }
    }
}
#endif // ALPS_NGS_ALEA_DETAIL_VALUE_TYPE_IMPLEMENTATION
