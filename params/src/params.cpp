#include <iostream>
#include <stdexcept>
//#include <vector>
//#include <algorithm>
//#include <iterator>
#include "boost/foreach.hpp"
// #include "boost/preprocessor.hpp"
#include "boost/algorithm/string/trim.hpp"
#include "boost/algorithm/string/erase.hpp"

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
        po::positional_options_description pd;
        pd.add(cfgfile_optname_,1); // FIXME: should it be "-1"? How the logic behaves for several positional options?
        po::parsed_options cmdline_opts=
            po::command_line_parser(argvec_).
            allow_unregistered().
            options(descr_).
            positional(pd).  
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
        BOOST_FOREACH(const params::value_type& pair, prm) {
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

}

namespace boost {
    namespace program_options {
        // Declaring validate() function in the boost::program_options namespace for it to be found by boost.
        void validate(boost::any& outval, const std::vector<std::string>& strvalues,
                      std::string* target_type, int)
        {
            namespace po=boost::program_options;
            namespace pov=po::validators;
            namespace alg=boost::algorithm;
            typedef std::vector<std::string> strvec;
            typedef boost::char_separator<char> charsep;
        
            pov::check_first_occurrence(outval); // check that this option has not yet been assigned
            std::string in_str=pov::get_single_string(strvalues); // check that this option is passed a single value

            // Now, do parsing:
            alg::trim(in_str); // Strip trailing and leading blanks
            // FIXME? Feature removed: if (in_str[in_str.size()-1]==';') alg::erase_tail(in_str,1); // Strip trailing semicolon
            if (in_str[0]=='"' && in_str[in_str.size()-1]=='"') { // Check if it is a "quoted string"
                // Strip surrounding quotes:
                alg::erase_tail(in_str,1);
                alg::erase_head(in_str,1);
                // FIXME? No special processing of the quotes inside the string.
            }
            outval=boost::any(in_str);
            std::cerr << "***DEBUG: returning from validate(...std::string*...) ***" << std::endl;
        }
    }
}
