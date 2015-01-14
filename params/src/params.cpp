#include <iostream>
#include <stdexcept>
#include "boost/foreach.hpp"
// #include "boost/preprocessor.hpp"
#include "boost/algorithm/string/join.hpp"

#include "alps/params.hpp"

/* Supported parameter types: */
#ifndef ALPS_PARAMS_SUPPORTED_TYPES
#define ALPS_PARAMS_SUPPORTED_TYPES (5, (int,unsigned,double,bool,std::string))
#endif

namespace po=boost::program_options;

namespace alps {

    const char* const params::cfgfile_optname_="parameter-file";
    
    params::params(hdf5::archive ar, std::string const & path)
    {
        throw std::logic_error("Not implemented yet");
    }

    void params::certainly_parse() const
    {
        po::parsed_options cmdline_opts=
            po::command_line_parser(argvec_).
            allow_unregistered().
            positional(po::positional_options_description().add(cfgfile_optname_,1)).  // FIXME: should it be "-1"? How the logic behaves for several positional options?
            options(descr_).
            run();

        varmap_=variables_map();
        po::store(cmdline_opts,*varmap_);

        if (varmap_->count(cfgfile_optname_) == 0) return;
        
        const std::string& cfgname=(*varmap_)[cfgfile_optname_].as<std::string>();
        po::parsed_options cfgfile_opts=
          po::parse_config_file<char>(cfgname.c_str(),descr_,true); // parse the file, allow unregistered options

        po::store(cfgfile_opts,*varmap_);
    }        

    void params::save(hdf5::archive& ar) const
    {
        throw std::logic_error("params::save() is not implemented yet");
    }

    void params::load(hdf5::archive& ar)
    {
        throw std::logic_error("params::load() is not implemented yet");
    }

    bool params::help_requested(std::ostream& ostrm)
    {
        if (varmap_->count("help")) {
            ostrm << helpmsg_ << std::endl;
            ostrm << descr_;
            return true;
        }
        return false;
    }        

        
    /// Output parameters to a stream
    std::ostream& operator<< (std::ostream& os, const params& prm)
    {
        BOOST_FOREACH(const params::value_type& pair, prm)
        {
            const std::string& k=pair.first;
            const params::mapped_type& v=pair.second;
            os << k << "=";
            // FIXME: the following game with iterators and assertions can be avoided
            //        if the printout function would be a member of params::mapped_type;
            //        however, it requires deriving from boost::variables_map.
            params::printout_map_type::const_iterator pit=prm.printout_map_.find(k);
            assert(pit != prm.printout_map_.end() && "Printout function is given for a parameter");
            (pit->second)(os,v.value());
            os << std::endl;
        }
        return os;
    }
  
    // FIXME:ToDo: file parsing, especially for lists (vectors)

}

