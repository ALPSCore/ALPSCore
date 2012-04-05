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


#ifndef ALPS_NGS_ALEA_DETAIL_VALUE_TYPE_ADAPTER_HEADER
#define ALPS_NGS_ALEA_DETAIL_VALUE_TYPE_ADAPTER_HEADER

#include <alps/ngs/alea/accumulator_impl.hpp>

#include <typeinfo>
namespace alps
{
    namespace alea
    {
        namespace detail
        {
            //setting up the dependencies for value_type-Adapter isn't neccessary bc has none
            
            template<typename T, typename base> 
            class Adapter<ValueType<T>, base>
            {
                typedef Adapter<ValueType<T>, base> ThisType;
                public:
                    typedef T value_type;
                    
                    Adapter<ValueType<T>, base>(ThisType const & arg): count_(arg.count_) {}
                    
                    template <typename ArgumentPack>
                    Adapter<ValueType<T>, base>(ArgumentPack const & args, typename boost::disable_if<
                                                                                                  boost::is_base_of<ThisType, ArgumentPack>
                                                                                                , int
                                                                                                >::type = 0
                                            ): count_() 
                    {}
                    
                    ThisType& operator <<(value_type val) 
                    {
                        ++count_;
                        return *this;
                    }
                    
                    boost::int64_t count() const 
                    { 
                        return count_; 
                    }
                
                    template<typename Stream> 
                    void print(Stream & os) 
                    {
                        os << "ValueType: " << typeid(value_type).name() << " ";
                        os << "Count: " << count() << " ";
                    }
                
                private:
                    boost::int64_t count_;
            };
        } // end namespace detail
    }//end alea namespace 
}//end alps namespace
#endif // ALPS_NGS_ALEA_DETAIL_VALUE_TYPE_ADAPTER
