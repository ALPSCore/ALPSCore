/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2001-2009 by Matthias Troyer <troyer@comp-phys.org>,
*                            Synge Todo <wistaria@comp-phys.org>
*
* This software is part of the ALPS libraries, published under the ALPS
* Library License; you can use, redistribute it and/or modify it under
* the terms of the license, either version 1 or (at your option) any later
* version.
* 
* You should have received a copy of the ALPS Library License along with
* the ALPS Libraries; see the file LICENSE.txt. If not, the license is also
* available from http://alps.comp-phys.org/.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
* FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT 
* SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE 
* FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE, 
* ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
* DEALINGS IN THE SOFTWARE.
*
*****************************************************************************/

/* $Id$ */

#ifndef ALPS_LATTICE_HYPERCUBIC_H
#define ALPS_LATTICE_HYPERCUBIC_H

#include <alps/config.h>
#include <alps/lattice/lattice.h>
#include <alps/lattice/coordinate_traits.h>

#include <alps/utility/vectorio.hpp>
#include <alps/numeric/vector_functions.hpp>
#include <alps/numeric/scalar_product.hpp>

#include <boost/utility.hpp>

#include <cmath>
#include <vector>

#ifndef M_PI
#define M_PI std::acos(-1.)
#endif

namespace alps {

template <class BASE, class EX = typename lattice_traits<BASE>::offset_type>
class hypercubic_lattice;

template <class BASE, class EX>
inline std::size_t dimension(const hypercubic_lattice<BASE,EX>& l)
{
  return l.dimension();
}


template <class BASE, class EX>
class hypercubic_lattice : public BASE {
public:
  typedef hypercubic_lattice<BASE,EX> lattice_type;
  typedef BASE parent_lattice_type;

  typedef typename lattice_traits<parent_lattice_type>::unit_cell_type unit_cell_type;
  typedef typename lattice_traits<parent_lattice_type>::cell_descriptor cell_descriptor;
  typedef typename lattice_traits<parent_lattice_type>::offset_type offset_type;
  typedef EX extent_type;
  typedef typename lattice_traits<parent_lattice_type>::basis_vector_iterator basis_vector_iterator;
  typedef typename lattice_traits<parent_lattice_type>::vector_type vector_type;
  typedef boundary_crossing boundary_crossing_type;
  typedef std::vector<unsigned int> distance_type;

  typedef int size_type;

  hypercubic_lattice() {}

  template <class BASE2, class EX2>
  hypercubic_lattice(const hypercubic_lattice<BASE2,EX2>& l)
   : parent_lattice_type(l), extent_(l.extent().begin(), l.extent().end()),
     bc_(l.boundary())
   { fill_extent(); }

  hypercubic_lattice(const parent_lattice_type& p, size_type length,
                     const std::string& bc = "periodic")
    : parent_lattice_type(p), extent_(BASE::dimension(), length),
      bc_(BASE::dimension(), bc) {}

  template <class InputIterator>
  hypercubic_lattice(const parent_lattice_type& p,
                     InputIterator first, InputIterator last,
                     const std::string& bc = "periodic")
    : parent_lattice_type(p), extent_(first, last), bc_(BASE::dimension(), bc)
  { fill_extent(); }

  template <class InputIterator2>
  hypercubic_lattice(const parent_lattice_type& p, size_type length,
                     InputIterator2 first2, InputIterator2 last2)
    : parent_lattice_type(p), extent_(BASE::dimension(), length),
      bc_(first2, last2)
  { fill_extent(); }

     template <class InputIterator, class InputIterator2>
  hypercubic_lattice(const parent_lattice_type& p,
                     InputIterator first, InputIterator last,
                     InputIterator2 first2, InputIterator2 last2)
   : parent_lattice_type(p), extent_(first, last), bc_(first2, last2)
  { fill_extent(); }

  template <class BASE2, class EX2>
  const hypercubic_lattice& operator=(const hypercubic_lattice<BASE2,EX2>& l)
  {
    static_cast<BASE&>(*this)=l;
    extent_=l.extent();
    bc_ = l.boundary();
    return *this;
  }

  class cell_iterator {
  public:
      typedef std::ptrdiff_t difference_type;
      typedef cell_descriptor value_type;
      typedef cell_descriptor* pointer;
      typedef cell_descriptor& reference;
      typedef std::forward_iterator_tag iterator_category;

    cell_iterator() {}
    cell_iterator(const lattice_type& l, const offset_type& o)
     : lattice_(&l), offset_(o) {}

    const cell_iterator& operator++() {
      typedef typename coordinate_traits<offset_type>::iterator offset_iterator;
      typedef typename coordinate_traits<offset_type>::const_iterator const_offset_iterator;
      offset_iterator offit, offend;
      boost::tie(offit,offend)=coordinates(offset_);
      int d=0;
      while ((offit+1)!=offend) {
        ++d;
        ++offit;
      }
      (*offit)++;
      while (*offit == static_cast<int>(lattice_->extent(d)) && d > 0) //go backwards
        {
          *offit =0;
          --d;
          --offit;
          (*offit)++;
        }
      return *this;
    }
    cell_iterator operator++(int) {
      cell_iterator tmp(*this);
      operator++();
      return tmp;
    }

    bool operator ==(const cell_iterator& it)
    {
      return (lattice_ == it.lattice_ && offset_ == it.offset_);
    }

    bool operator !=(const cell_iterator& it)
    {
      return (lattice_ != it.lattice_ || offset_ != it.offset_);
    }

    cell_descriptor operator*() const { return lattice_->cell(offset_);}
    // operator-> looks harder to implement. any good ideas?

  protected:
    const lattice_type* lattice_;
    offset_type offset_;
  };

  std::pair<cell_iterator,cell_iterator> cells() const
  {
    offset_type begin(extent_.begin(), extent_.end());
    offset_type end(extent_.begin(), extent_.end());
    std::fill(coordinates(begin).first,coordinates(begin).second,0);
    if (coordinates(end).first != coordinates(end).second)
      std::fill(coordinates(end).first+1,coordinates(end).second,0);
    else
      std::cerr << "Strange extent: " << write_vector(extent_) << "\n";
    return std::make_pair(cell_iterator(*this,begin),cell_iterator(*this,end));
  }

  size_type volume() const {
    return std::accumulate(extent_.begin(),extent_.end(),size_type(1),
           std::multiplies<size_type>());
  }


  size_type index(const cell_descriptor& c) const
  {
    size_type ind=0;
    size_type factor=1;
    offset_type o=offset(c,*this);
    typedef typename coordinate_traits<offset_type>::const_iterator OCI;
    typedef typename coordinate_traits<extent_type>::const_iterator ECI;
    ECI exit = coordinates(extent_).first;
    OCI offit = coordinates(o).first;
    if (exit == coordinates(extent_).second)
      return 0;
    int d=1;
    while(exit+1 != coordinates(extent_).second 
      && offit+1 != coordinates(o).second) {
      ++d;
      ++exit;
      ++offit;
    }

    while (true) {
      ind += factor* (*offit);
      factor *= static_cast<int>(*exit);
      --d;
      if (d == 0) break;
      --exit;
      --offit;
    }
    return ind;
  }

  bool on_lattice(const cell_descriptor& c) const {
    typedef typename coordinate_traits<offset_type>::const_iterator OCI;
    typedef typename coordinate_traits<extent_type>::const_iterator ECI;
    ECI exit = coordinates(extent_).first;
    OCI offit = coordinates(offset(c,*this)).first;

    for (; exit != coordinates(extent_).second; ++exit, ++offit)
      if(*offit<0 || *offit>=static_cast<int>(*exit))
        return false;
    return true;
  }

  cell_descriptor cell(size_type i)  const{
    offset_type o(extent_.begin(), extent_.end());
    typedef typename coordinate_traits<extent_type>::const_iterator CIT;
    typedef typename coordinate_traits<offset_type>::iterator IT;
    CIT ex = coordinates(extent_).first;
    IT offit = coordinates(o).first;
    
    //changed order of dimensions
    for (;ex!=coordinates(extent_).second;++ex,++offit) {};
    
    do {
      --ex; 
      --offit;
      *offit=i%static_cast<int>(*ex);
      i/=static_cast<int>(*ex); }    
    while ( ex!=coordinates(extent_).first);
    
    //replaced this code:    
    //for (;ex!=coordinates(extent_).first;--ex,--offit) {
    //  *offit=i%(*ex);
    //  i/=(*ex);
    // }
    return cell(o);
  }

  cell_descriptor cell(offset_type o) const
  {
    return alps::cell(o,static_cast<const parent_lattice_type&>(*this));
  }

  std::pair<bool,boundary_crossing_type> shift(offset_type& o,const offset_type& s) const
  {
    using boost::numeric::operators::operator+;
    o=o+s;
    typedef typename coordinate_traits<offset_type>::iterator IT;
    typedef typename coordinate_traits<extent_type>::const_iterator CIT;
    IT offit=alps::coordinates(o).first;
    CIT exit=alps::coordinates(extent_).first;
    std::vector<std::string>::const_iterator bit=bc_.begin();
    boundary_crossing_type crossing;
    for (int dim=0; exit!=alps::coordinates(extent_).second;++dim, ++bit, ++offit, ++exit) {
      if (*offit<0)
        while (*offit<0) {
                  if (*bit=="periodic") {
                    *offit+=static_cast<int>(*exit); // need to check % for negative numbers
            crossing.set_crossing(dim,-1);
          }
                  else
                    return std::make_pair(false,boundary_crossing_type());
        }
      else if (*offit >= static_cast<int>(*exit)) {
                if (*bit=="periodic") {
                  *offit %= static_cast<int>(*exit);
          crossing.set_crossing(dim,1);
        }
                else
                  return std::make_pair(false,boundary_crossing_type());
      }
    }
    return std::make_pair(true,crossing);
  }

  const std::string& boundary(unsigned int dim) const{ return bc_[dim];}
  const std::vector<std::string>& boundary() const { return bc_;}
  typename extent_type::value_type extent(unsigned int dim) const {return extent_[dim];}
  const extent_type& extent() const { return extent_;}


  std::vector<std::string> distance_labels(int precision = 0) const
  {
    std::vector<std::string> label(num_distances());
    for (cell_iterator it1=cells().first; it1 != cells().second;++it1) {
      for (cell_iterator it2=cells().first; it2 != cells().second;++it2) {
        offset_type x=alps::offset(*it1,*this);
        offset_type y=alps::offset(*it2,*this);
        std::size_t d=distance(alps::offset(*it1,*this),alps::offset(*it2,*this));
        if (label[d].empty())
          label[d] =
            alps::coordinate_to_string(alps::offset(*it1,*this), precision)+" -- " + 
            alps::coordinate_to_string(alps::offset(*it2,*this), precision);
      }
    }
    return label;
  }

  std::vector<std::string> momenta_labels(int precision = 0) const
  {
    std::vector<std::string> label;
    for (momentum_iterator it=momenta().first; it != momenta().second;++it)
      label.push_back(alps::coordinate_to_string(*it, precision)); 
    return label;
  }
  
  std::vector<unsigned int> distance_multiplicities() const
  {
    std::vector<unsigned int> mult(num_distances());
    for (cell_iterator it1=cells().first; it1 != cells().second; ++it1)
      for (cell_iterator it2=cells().first; it2 != cells().second; ++it2)
        mult[distance(alps::offset(*it1,*this),alps::offset(*it2,*this))]++;
    return mult;
  }

  std::size_t num_distances() const
  {
    std::size_t d=1;
    for (unsigned int i=0;i<BASE::dimension();++i) {
      if(boundary(i)=="periodic")
        d*=static_cast<std::size_t>(extent(i));
      else
        d*=static_cast<std::size_t>(extent(i))*static_cast<std::size_t>(extent(i));
    }
    return d;
  }


  std::size_t distance(const offset_type& x, const offset_type& y) const
    {
      std::size_t d=0;
      for (unsigned int i=0;i<BASE::dimension();++i) {
        if(boundary(i)=="periodic")
          d = d*static_cast<std::size_t>(extent(i)) +
            (x[i] <= y[i] ? y[i]-x[i] : static_cast<std::size_t>(extent(i))+y[i]-x[i]);
        else
          d = static_cast<std::size_t>(extent(i))*(d*static_cast<std::size_t>(extent(i)) +x[i])+y[i];
      }
      return d;
    }


  class momentum_iterator : public cell_iterator {
  public:
    momentum_iterator(cell_iterator it=cell_iterator()) : cell_iterator(it), valid_(false) {}
    const vector_type& operator*() const { set_k(); return k_; }
    const vector_type* operator->() const { set_k(); return &k_; }
    std::complex<double> phase(const vector_type& pos) const {
      set_k();
      double phase=numeric::scalar_product(k_,pos);
      return std::complex<double>(std::cos(phase),std::sin(phase));
    }
    const momentum_iterator& operator++() { 
      valid_=false; 
      cell_iterator::operator++(); 
      return *this;
    }
    
    const momentum_iterator& operator++(int) { 
      momentum_iterator save(*this);
      operator++();
      return save;
    }

  private:
    mutable vector_type k_;
    mutable bool valid_;
    void set_k() const
    {
      if (valid_)
        return;
      valid_=true;
      k_.resize(alps::dimension(*cell_iterator::lattice_));
      for (unsigned int i=0;i<alps::dimension(*cell_iterator::lattice_);++i) {
        k_[i] = 0.;
        int l=cell_iterator::lattice_->extent()[i];
        int x=cell_iterator::offset_[i];
        if (x>l/2)
          x-=l;
        k_[i] = 2.*M_PI*double(x)/double(l);
      }
    }
  };

  std::pair<momentum_iterator,momentum_iterator> momenta() const
  {
    return std::make_pair(momentum_iterator(cells().first),momentum_iterator(cells().second));
  }

  std::vector<int> translation_directions() const 
  {
    std::vector<int> dirs;
    for (std::size_t i=0;i<bc_.size();++i)
      if (bc_[i]=="periodic")
        dirs.push_back(i);
    return dirs;
  }

  std::vector<vector_type> translation_momenta() const 
  {
    std::vector<vector_type> ks;
    for (std::size_t i=0;i<bc_.size();++i)
      if (bc_[i]=="periodic") {
        if (ks.empty())
          for (int j = 0; j < static_cast<int>(extent_[i]); ++j)
            ks.push_back(vector_type(1,2.*j*M_PI/ static_cast<int>(extent_[i])));
        else {
          std::vector<vector_type> newks;
          for (std::size_t l=0;l<ks.size();++l) 
            for (int j = 0; j < static_cast<int>(extent_[i]); ++j) {
              vector_type k=ks[l];
              k.push_back(2.*j*M_PI/ static_cast<int>(extent_[i]));
              newks.push_back(k);
            }
          ks.swap(newks);
        }
      }
    return ks;
  }
  
  std::vector<std::pair<std::complex<double>,std::vector<std::size_t> > > translations(const vector_type& k) const
  {
    std::vector<int> dirs=translation_directions();
    std::vector<std::pair<std::complex<double>,std::vector<std::size_t> > > trans;
    if (alps::dimension(k)!=dirs.size())
      boost::throw_exception(std::runtime_error("Incorrect number of momenta specified in hypercubic_lattice::translations"));
    
    std::vector<int> theshift(dirs.size());
    offset_type off(extent_.begin(), extent_.end());
    for (std::size_t i=0;i<alps::dimension(off);++i)
      off[i]=0;
    
    bool done=false;
    while (!done) {
      double phase=0.;
      for (std::size_t i=0;i<dirs.size();++i) {
        off[dirs[i]]=theshift[i];
        phase += theshift[i] * static_cast<double>(k[i]);
      }
      std::vector<std::size_t> shifted_index;
      for (cell_iterator it=cells().first; it !=cells().second;++it) {
        offset_type shifted_offset=(*it).offset();
        if (!shift(shifted_offset,off).first)
          boost::throw_exception(std::logic_error("Shifting along periodic direction leaves lattice"));
        shifted_index.push_back(index(cell(shifted_offset)));
      }
      trans.push_back(std::make_pair(std::exp(std::complex<double>(0.,phase)),shifted_index));
      done=true;
      for (std::size_t i=0; i<theshift.size() && done;++i) {
        if (++theshift[i]>=static_cast<int>(extent_[dirs[i]])) 
          theshift[i]=0;
         else
           done=false;
      }
    }
    
    return trans;
  }

protected:
  extent_type extent_;
  std::vector<std::string> bc_;
private:
  void fill_extent() {
    extent_.resize(BASE::dimension(),1);
    bc_.resize(BASE::dimension(),"open");
  }
};

template <class BASE, class EX>
struct lattice_traits<hypercubic_lattice<BASE,EX> >
{
  typedef typename hypercubic_lattice<BASE,EX>::unit_cell_type unit_cell_type;
  typedef typename hypercubic_lattice<BASE,EX>::cell_descriptor cell_descriptor;
  typedef typename hypercubic_lattice<BASE,EX>::offset_type offset_type;
  typedef typename hypercubic_lattice<BASE,EX>::extent_type extent_type;
  typedef typename hypercubic_lattice<BASE,EX>::basis_vector_iterator basis_vector_iterator;
  typedef typename hypercubic_lattice<BASE,EX>::momentum_iterator momentum_iterator;
  typedef typename hypercubic_lattice<BASE,EX>::cell_iterator cell_iterator;
  typedef typename hypercubic_lattice<BASE,EX>::size_type size_type;
  typedef typename hypercubic_lattice<BASE,EX>::vector_type vector_type;
  typedef typename hypercubic_lattice<BASE,EX>::boundary_crossing_type boundary_crossing_type;
};

} // end namespace alps

#endif // ALPS_LATTICE_HYPERCUBIC_H
