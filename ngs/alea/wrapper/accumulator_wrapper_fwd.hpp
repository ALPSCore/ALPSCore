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


#ifndef ALPS_NGS_ALEA_ACCUMULATOR_WRAPPER_FWD_HEADER
#define ALPS_NGS_ALEA_ACCUMULATOR_WRAPPER_FWD_HEADER

#include <boost/shared_ptr.hpp>
#include <boost/cstdint.hpp>
#include <alps/ngs/alea/accumulator.hpp>

#ifdef ALPS_HAVE_MPI
    #include <alps/ngs/boost_mpi.hpp>
#endif

namespace alps
{
    namespace accumulator
    {
        namespace detail
        {
            class base_wrapper;
            template<typename Accum>
            class result_type_wrapper;

            //class that holds the base_wrapper pointer
            class accumulator_wrapper {
                public:
                    template<typename T> 
                    accumulator_wrapper(T arg);
                    
                    accumulator_wrapper(accumulator_wrapper const & arg);
                    
                    template<typename T>
                    accumulator_wrapper& operator<<(const T& value);
                        
                    
                    template<typename T>
                    detail::result_type_wrapper<T> & get() const;//TODO
                    
                    friend std::ostream& operator<<(std::ostream &out, const accumulator_wrapper& wrapper);
                    
                    template <typename T>
                    T & extract() const;
                    
                    boost::uint64_t count() const;
                    
                    inline void reset();

#ifdef ALPS_HAVE_MPI
                    inline void collective_merge(
                          boost::mpi::communicator const & comm
                        , int root
                    );
                    inline void collective_merge(
                          boost::mpi::communicator const & comm
                        , int root
                    ) const;
#endif
                private:
                    boost::shared_ptr<base_wrapper> base_;
            };
        }//end detail namespace 
    }//end accumulator namespace 
}//end alps namespace
#endif // ALPS_NGS_ALEA_ACCUMULATOR_WRAPPER_FWD_HEADER

