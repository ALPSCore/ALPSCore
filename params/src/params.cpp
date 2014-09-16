/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#include <alps/params.hpp>
#include <boost/bind.hpp>
#include <boost/algorithm/string.hpp>
#include <algorithm>
#include <fstream>

namespace alps {
    params::params(hdf5::archive ar, std::string const & path) {
        std::string context = ar.get_context();
        ar.set_context(path);
        load(ar);
        ar.set_context(context);
    }

    params::params(boost::filesystem::path const & path) {
      parse_text_parameters(path);
    }

    #ifdef ALPS_HAVE_PYTHON
        params::params(boost::python::dict const & arg) {
            boost::python::extract<boost::python::dict> dict(arg);
            if (!dict.check())
                throw std::invalid_argument("parameters can only be created from a dict" + ALPS_STACKTRACE);
            const boost::python::object kit = dict().iterkeys();
            const boost::python::object vit = dict().itervalues();
            for (std::size_t i = 0; i < boost::python::len(dict()); ++i)
                setter(boost::python::call_method<std::string>(kit.attr("next")().ptr(), "__str__"), vit.attr("next")());
        }
    #endif

    std::size_t params::size() const {
        return keys.size();
    }

    void params::erase(std::string const & key) {
        if (!defined(key))
            throw std::invalid_argument("the key " + key + " does not exists" + ALPS_STACKTRACE);
        keys.erase(find(keys.begin(), keys.end(), key));
        values.erase(key);
    }

    params::value_type params::operator[](std::string const & key) {
        return value_type(
            defined(key),
            boost::bind(&params::getter, boost::ref(*this), key),
            boost::bind(&params::setter, boost::ref(*this), key, _1),
            key
        );
    }

    params::value_type const params::operator[](std::string const & key) const {
        return defined(key)
            ? value_type(values.find(key)->second, key)
            : value_type(key)
        ;
    }

    bool params::defined(std::string const & key) const {
        return values.find(key) != values.end();
    }

    params::iterator params::begin() {
        return iterator(*this, keys.begin());
    }

    params::const_iterator params::begin() const {
        return const_iterator(*this, keys.begin());
    }

    params::iterator params::end() {
        return iterator(*this, keys.end());
    }

    params::const_iterator params::end() const {
        return const_iterator(*this, keys.end());
    }

    void params::save(hdf5::archive & ar) const {
        for (params::const_iterator it = begin(); it != end(); ++it)
            ar[it->first] << it->second;
    }

    void params::load(hdf5::archive & ar) {
        keys.clear();
        values.clear();
        std::vector<std::string> list = ar.list_children(ar.get_context());
        for (std::vector<std::string>::const_iterator it = list.begin(); it != list.end(); ++it) {
            detail::paramvalue value;
            ar[*it] >> value;
            setter(*it, value);
        }
    }

    #ifdef ALPS_HAVE_MPI
        void params::broadcast(boost::mpi::communicator const & comm, int root) {
            boost::mpi::broadcast(comm, *this, root);
        }
    #endif
    
    void params::setter(std::string const & key, detail::paramvalue const & value) {
        if (!defined(key)){
            keys.push_back(key);
        }
        values[key] = value;
    }

    detail::paramvalue params::getter(std::string const & key) {
        return values[key];
    }
    

    std::ostream & operator<<(std::ostream & os, params const & v) {
        for (params::const_iterator it = v.begin(); it != v.end(); ++it)
            os << it->first << " = " << it->second << std::endl;
        return os;
    }
  //this is a trivial parameter parser to handle text parameters
  //we expect parameters to be of the form key = value
  //format taken over from legacy ALPS.
  void params::parse_text_parameters(boost::filesystem::path const & path){
    keys.clear();
    values.clear();
    std::ifstream ifs(path.string().c_str());
    if(!ifs.is_open()) throw std::runtime_error("Problem reading parameter file at: "+path.string());
    
    std::string line;
    while(std::getline(ifs, line)){
      std::size_t eqpos=line.find("=");
      if (eqpos==std::string::npos || eqpos ==0 || eqpos ==line.length()) continue; //no equal sign found in this line or string starting/ending with =
      std::string key=line.substr(0,eqpos);
      std::string value=line.substr(eqpos+1,line.length());
      boost::algorithm::trim(key);
      boost::algorithm::trim(value);
      boost::algorithm::trim_if(value, boost::is_any_of(";")); //trim semicolon
      setter(key,value);
    }
  }
}
