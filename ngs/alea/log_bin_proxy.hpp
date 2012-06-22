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

#ifndef ALPS_NGS_ALEA_LOG_BIN_PROXY_HEADER
#define ALPS_NGS_ALEA_LOG_BIN_PROXY_HEADER

#include <vector>
#include <ostream>
#include <cmath>
#include <algorithm>


#include <alps/ngs/alea/mean_type_trait.hpp>

namespace alps
{
    namespace alea
    {
        template<typename value_type>
        class log_bin_proxy_type
        {
            typedef typename mean_type<value_type>::type mean_type;
            typedef typename std::vector<value_type>::size_type size_type;
            static std::vector<mean_type> unused;
        public:
            log_bin_proxy_type(): bin_(unused) {}
            log_bin_proxy_type(std::vector<mean_type> const & bin): bin_(bin) {}
            
            inline std::vector<mean_type> const & bins() const 
            {
                return bin_;
            }
            
            template<typename T>
            friend std::ostream & operator<<(std::ostream & os, log_bin_proxy_type<T> const & arg);
        private:
            std::vector<mean_type> const & bin_;
        };

        //~ template<typename value_type>
        //~ log_bin_proxy_type<value_type> make_log_bin_proxy_type()
        //~ {
            //~ return log_bin_proxy_type<value_type>(unused);
        //~ }
        
        template<typename T>
        inline std::ostream & operator<<(std::ostream & os, log_bin_proxy_type<T> const & arg)
        {
            os << "log_bin_proxy" << std::endl;
            return os;
            
        };
    }//end alea namespace 
}//end alps namespace

#endif //ALPS_NGS_ALEA_LOG_BIN_PROXY_HEADER
