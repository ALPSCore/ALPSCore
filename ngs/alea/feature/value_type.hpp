/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2011 - 2012 by Mario Koenz <mkoenz@ethz.ch>                       *
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


#ifndef ALPS_NGS_ALEA_DETAIL_VALUE_TYPE_IMPLEMENTATION_HEADER
#define ALPS_NGS_ALEA_DETAIL_VALUE_TYPE_IMPLEMENTATION_HEADER

#include <alps/ngs/alea/accumulator/accumulator_impl.hpp>
#include <boost/cstdint.hpp>
#include <typeinfo>

#ifdef ALPS_HAVE_MPI
    #include <alps/ngs/boost_mpi.hpp>
#endif

namespace alps
{
    namespace accumulator
    {
        //=================== value_type trait ===================
        template <typename Accum>
        struct value_type
        {
            typedef typename Accum::value_type type;
        };
        //=================== value_type implementation ===================
        namespace detail
        {
            //setting up the dependencies for value_type-Implementation isn't neccessary bc has none
            
            template<typename T, typename base_type> 
            class Implementation<ValueType<T>, base_type>
            {
                typedef Implementation<ValueType<T>, base_type> ThisType;
                public:
                    typedef T value_type;
                    
                    Implementation<ValueType<T>, base_type>(ThisType const & arg): count_(arg.count_) {}
                    
                    template <typename ArgumentPack>
                    Implementation<ValueType<T>, base_type>(ArgumentPack const & args, typename boost::disable_if<
                                                                                                  boost::is_base_of<ThisType, ArgumentPack>
                                                                                                , int
                                                                                                >::type = 0
                                            ): count_() 
                    {}
                    
                    inline ThisType& operator <<(value_type val) 
                    {
                        ++count_;
                        return *this;
                    }
                    
                    inline boost::uint64_t const & count() const 
                    { 
                        return count_; 
                    }
                
                    template<typename Stream> 
                    inline void print(Stream & os) 
                    {
                        os << "ValueType: " << typeid(value_type).name() << " " << std::endl;
                        os << "Count: " << count() << " " << std::endl;
                    }
                    inline void reset()
                    {
                        count_ = 0;
                    }

#ifdef ALPS_HAVE_MPI
                    void collective_merge(
                          boost::mpi::communicator const & comm
                        , int root
                    ) {
                        if (comm.rank() == root)
                            boost::mpi::reduce(comm, count_, count_, std::plus<boost::uint64_t>(), root);
                        else
                            const_cast<ThisType const *>(this)->collective_merge(comm, root);
                    }
                    void collective_merge(
                          boost::mpi::communicator const & comm
                        , int root
                    ) const {
                        if (comm.rank() == root)
                            throw std::runtime_error("this object is const" + ALPS_STACKTRACE);

                        else
                            boost::mpi::reduce(comm, count_, std::plus<boost::uint64_t>(), root);
                    }
#endif                    
                private:
                    boost::uint64_t count_;
            };
        } // end namespace detail
    }//end accumulator namespace 
}//end alps namespace
#endif // ALPS_NGS_ALEA_DETAIL_VALUE_TYPE_IMPLEMENTATION
