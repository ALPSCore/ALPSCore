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
