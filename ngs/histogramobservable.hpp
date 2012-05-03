/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2010 - 2011 by Lukas Gamper <gamperl@gmail.com>                   *
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


#ifndef ALPS_NGS_HISTOGRAMOBSERVABLE_HPP
#define ALPS_NGS_HISTOGRAMOBSERVABLE_HPP

#include <alps/ngs/hdf5.hpp>
#include <alps/ngs/hdf5/vector.hpp>


namespace alps {
   namespace ngs {

       template<class T>
       class histogram_observable
       {
       public:
           typedef T value_type;
           typedef std::vector<value_type> container_type;
           typedef typename container_type::size_type size_type;

           histogram_observable(const std::string& name="", size_type size=0) : name_(name), values_(size) {};
           
           const std::string& name() const { return name_; }
           size_type size() const { return values_.size(); }

           histogram_observable& operator<<(const std::pair<size_type,value_type>& v)   { values_[v.first] += v.second; return *this; }
           value_type& operator[](size_type i)  { return values_[i]; }
           value_type operator[](size_type i) const { return values_[i]; }

           void save(hdf5::archive & ar) const  { ar << make_pvp("name",name_) << make_pvp("values",values_); }
           void load(hdf5::archive & ar)        { ar >> make_pvp("name",name_) >> make_pvp("values",values_); }

       private:
           std::string name_;
           container_type values_;
       };

   }
}

#endif
