/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2011 by Matthias Troyer                                           *
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

#ifndef ALPS_NGS_OBSERVABLE_HPP
#define ALPS_NGS_OBSERVABLE_HPP

#include <boost/shared_ptr.hpp>
#include <iostream>

namespace alps {
  
  namespace detail {

    class observable_base
    {
    public:
      observable_base() {}
      virtual std::string name() const =0;
      virtual void output(sd::ostream&) const =0;
      virtual void save(hdf5::archive & ar) const =0;
      virtual void load(hdf5::archive & ar) =0;
      virtual void operator<<(boost::any const&) =0;
      virtual boost::any mean() const =0;
    };
    
    template <class Observable>
    class observable_impl: public observable_base
    {
    public:
      std::string name() const { return obs.name();}
      void output(sd::ostream& os) const { os << obs;}

      void save(hdf5::archive & ar) const { obs.save(ar);}
      void load(hdf5::archive & ar) { obs.load(ar);}

      void operator<<(boost::any const& x)
      {
        obs << boost::any_cast<typename Observable::value_type>(x);  
      }
  
      virtual boost::any mean() const { return boost::any(obs.mean());}
    private:
      Observable obs;
    };
    
  }
  
  class observable {

  public:

    /*
    mcobservable();
    mcobservable(Observable const * obs);
    mcobservable(mcobservable const & rhs);

    virtual ~mcobservable();

    mcobservable & operator=(mcobservable rhs);

    Observable * get();

    Observable const * get_impl() const;

    template<typename T> mcobservable & operator<<(T const & value);

    void save(hdf5::archive & ar) const;
    void load(hdf5::archive & ar);

    void merge(mcobservable const &);

    void output(std::ostream & os) const;
*/
  private:

    boost::shared_ptr<detail::observable_base> impl;

  };

  inline std::ostream & operator<<(std::ostream & os, observable const & obs)
  {
    obs.output(os);
    return os;
  }
  
//  Observable: impl ..
  
}

}

#endif
