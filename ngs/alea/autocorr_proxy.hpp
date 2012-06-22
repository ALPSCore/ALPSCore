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

#ifndef ALPS_NGS_ALEA_AUTOCORR_PROXY_HEADER
#define ALPS_NGS_ALEA_AUTOCORR_PROXY_HEADER

#include <vector>
#include <ostream>
#include <cmath>
#include <algorithm>


#include <boost/cstdint.hpp>

#include <alps/ngs/alea/mean_type_trait.hpp>

namespace alps
{
    namespace alea
    {
        template<typename value_type>
        class autocorr_proxy_type
        {
            typedef typename mean_type<value_type>::type mean_type;
            static std::vector<value_type> unused;
            static boost::int64_t unused2;
        public:
            autocorr_proxy_type(): bin2_(unused)
                                , bin1_(unused)
                                , count_(unused2) {}
            autocorr_proxy_type(  std::vector<value_type> const & bin2
                                , std::vector<value_type> const & bin1
                                , boost::uint64_t const & count):
                                                                  bin2_(bin2)
                                                                , bin1_(bin1)
                                                                , count_(count)
            {}
            
            inline std::vector<value_type> const & bins() const 
            {
                return bin2_;
            }
            inline std::vector<value_type> const & sum() const 
            {
                return bin1_;
            }
            
            inline mean_type const error(boost::uint64_t level = -1) const
            {
                if(level == -1)
                    level = std::max(bin2_.size() - 5, typename std::vector<value_type>::size_type(0));
                using std::sqrt;
                return sqrt((bin2_[level] - bin1_[level] * bin1_[level]) / (count_ - 1));
            }
            
            template<typename T>
            friend std::ostream & operator<<(std::ostream & os, autocorr_proxy_type<T> const & arg);
        private:
            std::vector<value_type> const & bin2_;
            std::vector<value_type> const & bin1_;
            boost::uint64_t const & count_;
        };

        template<typename T>
        inline std::ostream & operator<<(std::ostream & os, autocorr_proxy_type<T> const & arg)
        {
            os << "autocorr_proxy" << std::endl;
            return os;
            
        };
    }//end alea namespace 
}//end alps namespace

#endif //ALPS_NGS_ALEA_AUTOCORR_PROXY_HEADER
