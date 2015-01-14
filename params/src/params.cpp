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
        // if (argvec_.empty()) return;
        
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

    // FIXME: should it be moved to a private header? Or hidden as static or in private namespace?
    namespace {

        /// Service function: output a sequence
        template <typename T>
        std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec)
        {
            typedef std::vector<T> VT;
            if (vec.empty()) return os;
            typename VT::const_iterator it=vec.begin();
            typename VT::const_iterator end=vec.end();
            os << *it; // FIXME: possible stream errors ignored!
            ++it;
            for (; it!=end; ++it) {
                os << "," << *it;
            }
            return os;
        }
        
        /// output function for a boost::any holding a known type T; @returns true if the type is guessed right.
        template <typename T>
        bool printout(std::ostream& os, const boost::any& a) {
            typedef T value_type;
            const value_type* val=boost::any_cast<value_type>(&a);
            if (!val) return false;
            os << *val;
            return true;
        }

        /// output function for a boost::any holding std::string; @returns true if the type is guessed right.
        template <>
        bool printout<std::string>(std::ostream& os, const boost::any& a) {
            typedef std::string value_type;
            const value_type* val=boost::any_cast<value_type>(&a);
            if (!val) return false;
            os << "'" << *val << "'";
            return true;
        }

    } // end anonymous namespace
        
    
//     /// Output operator for a known boost::any
//     std::ostream& operator<< (std::ostream& os, const boost::any& a)
//     {
//         // FIXME: this is UGLY!
//         // Logic: we try each type; if the conversion works, we output it, else try next.
//         // If none works, we throw.
// #define LOCAL_OUTPUT_AS(_r_,a,T) if (const T* tmp=boost::any_cast<T>(&a)) { os << *tmp; } else
//         BOOST_PP_SEQ_FOR_EACH(LOCAL_OUTPUT_AS, a, BOOST_PP_TUPLE_TO_SEQ(ALPS_PARAMS_SUPPORTED_TYPES))
//         {
//             std::cerr << "Cannot output this type: " << a.type().name() << std::endl;
//             throw std::logic_error(std::string("Cannot output this type: ")+a.type().name());
//         }
// #undef LOCAL_OUTPUT_AS
//         return os;
//     }
            
    /// Output operator for a known boost::any
    std::ostream& operator<< (std::ostream& os, const boost::any& a)
    {
        // FIXME: this is UGLY!
        // Logic: we try to output each type; if it does not work, we try next.
        // If none works, we throw.
        if (!(
                   printout<int>(os,a)
                || printout<unsigned>(os,a)
                || printout<double>(os,a)
                || printout<bool>(os,a)
                || printout<std::string>(os,a)
                || printout< std::vector<unsigned> >(os,a)
                || printout< std::vector<double> >(os,a)
                )) {
            throw std::logic_error(std::string("Cannot output this type with typeid=")+a.type().name());
        }
        return os;
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

