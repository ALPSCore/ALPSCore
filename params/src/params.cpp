#include <iostream>
#include <sstream>
#include <stdexcept>
//#include <vector>
//#include <algorithm>
//#include <iterator>
#include "boost/foreach.hpp"
// #include "boost/preprocessor.hpp"
#include "boost/algorithm/string/trim.hpp"
#include "boost/algorithm/string/erase.hpp"
#include "boost/algorithm/string/classification.hpp"
#include "boost/algorithm/string/predicate.hpp"

#include "boost/optional.hpp"

// Serialization headers:
#include "boost/archive/text_oarchive.hpp"
#include "boost/archive/text_iarchive.hpp"

#include <alps/hdf5/archive.hpp>

#include "alps/params.hpp"

// /* Supported parameter types: */
// #ifndef ALPS_PARAMS_SUPPORTED_TYPES
// #define ALPS_PARAMS_SUPPORTED_TYPES (5, (int,unsigned,double,bool,std::string))
// #endif


// Anonymous namespace for service functions & classes
namespace {
    // Service functor class to convert C-string pointer to an std::string
    struct cstr2string {
        std::string operator()(const char* cstr)
        {
            return std::string(cstr);
        }
    };

    // Service function to try to open an HDF5 archive, return "none" if it fails
    boost::optional<alps::hdf5::archive> try_open_ar(const std::string& fname, const char* mode)
    {
        try {
            return alps::hdf5::archive(fname, mode);
        } catch (alps::hdf5::archive_error& ) {
            return boost::none;
        }
    };
}

namespace alps {
    namespace params_ns {
  
        namespace po=boost::program_options;
    
        /// Constructor from command line and a parameter file. The parsing is deferred.
        /** Tries to see if the file is an HDF5, in which case restores the object from the
            HDF5 file, ignoring the command line.
            @param hdfpath : path to HDF5 dataset containing the saved parameter object
            (0 if this functionality is not needed)
        */
        params::params(unsigned int argc, const char* argv[], const char* hdfpath)
        {
            if (argc>0) argv0_=argv[0];
            if (argc>1) {
                if (argv[1][0]!='-') {
                    // first argument exists and is not an option
                    infile_=argv[1];
                    if (hdfpath) {
                        boost::optional<alps::hdf5::archive> ar=try_open_ar(infile_, "r");
                        if (ar) {
                            this->load(*ar, hdfpath);
                            archname_=argv[1];
                            return; // nothing else to be done
                        }
                    }
                    // skip the first argument
                    --argc;
                    ++argv;
                }
                // save the command line
                std::transform(argv+1,argv+argc, std::back_inserter(argvec_), cstr2string());
            }
            init();
        }

        /** Access a parameter: read-only */
        const params::mapped_type& params::operator[](const std::string& k) const
        {
            possibly_parse();
            return optmap_[k];
        }

        /** Access a parameter: possibly for assignment */
        params::mapped_type& params::operator[](const std::string& k)
        {
            possibly_parse();
            return optmap_[k];
        }


        /// Function to check for validity/redefinition of an option (throws!)
        void params::check_validity(const std::string& optname) const
        {
            if ( ! (boost::all(optname, boost::is_alnum()||boost::is_any_of("_.-"))
                    && boost::all(optname.substr(0,1), boost::is_alnum()) ))  {
                // The name is not alphanum-underscore-dot-dash, or does not start with alnum
                throw invalid_name(optname, "Invalid parameter name");
            }
            if (descr_map_.count(optname)) {
                // The option was already defined
                throw double_definition(optname,"Attempt to define already defined parameter");
            }
            if (optmap_.count(optname)) {
                // The option was already explicitly assigned or defined
                throw extra_definition(optname, "Attempt to define explicitly assigned parameter");
            }
        }
        
        
        void params::certainly_parse(po::options_description& odescr) const
        {
            // First, create program_options::options_description from the description map
            BOOST_FOREACH(const detail::description_map_type::value_type& kv, descr_map_)
            {
                kv.second.add_option(odescr, kv.first);
            }

            // Second, parse the parameters according to the description into the variables_map
            po::variables_map vm;

            // Parse even if the commandline is empty, to set default values.
            {
                po::parsed_options cmdline_opts=
                    po::command_line_parser(argvec_).
                    allow_unregistered().
                    options(odescr).
                    run();

                po::store(cmdline_opts,vm);
            }

            if (!infile_.empty()) {
                po::parsed_options cfgfile_opts=
                    po::parse_config_file<char>(infile_.c_str(),odescr,true); // parse the file, allow unregistered options

                po::store(cfgfile_opts,vm);
            }

            // Now for each defined option, copy the corresponding parsed option to the this's option map
            // NOTE#1: If file has changed since the last parsing, option values will NOT be reassigned!
            // (only options that are not yet in optmap_ are affected here,
            // to avoid overwriting an option that was assigned earlier.)
            // NOTE#2: The loop is over the content of the define()'d options (descr_map_)
            // so that options that are defined but are not in the command line and are without default will be set:
            // it is needed for "trigger" options. It may also be an opportunity to distinguish between
            // options that are define()'d but are missing and those which were never even define()'d.
            BOOST_FOREACH(const detail::description_map_type::value_type& slot, descr_map_) {
                const std::string& k=slot.first;
                const detail::description_map_type::mapped_type& dscval=slot.second;
                if (optmap_.count(k)) continue; // skip the keys that are already there
                dscval.set_option(optmap_[k], vm[k].value());
            }
            is_valid_=true;
        }        

        void params::save(hdf5::archive& ar) const
        {
            std::ostringstream outs; 
            {
                boost::archive::text_oarchive boost_ar(outs);
                boost_ar << *this;
            }
            ar["alps::params"] << outs.str();
        }

        void params::save(hdf5::archive& ar, const std::string& path) const
        {
            std::string context = ar.get_context();
            ar.set_context(path);
            save(ar);
            ar.set_context(context);
        }
            
        void params::load(hdf5::archive& ar)
        {
            std::string buf;
            ar["alps::params"] >> buf;
            std::istringstream ins(buf);
            {
                boost::archive::text_iarchive boost_ar(ins);
                boost_ar >> *this;
            }
        }

        void params::load(hdf5::archive& ar, const std::string& path)
        {
            std::string context = ar.get_context();
            ar.set_context(path);
            load(ar);
            ar.set_context(context);
        }

        bool params::help_requested(std::ostream& ostrm)
        {
            po::options_description odescr;
            certainly_parse(odescr);
            if (optmap_["help"]) { // FIXME: will conflict with explicitly-assigned "help" parameter
                ostrm << helpmsg_ << std::endl;
                ostrm << odescr;
                return true;
            }
            return false;
        }        

    } // params_ns
} // alps

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
            if (in_str[0]=='"' && in_str[in_str.size()-1]=='"') { // Check if it is a "quoted string"
                // Strip surrounding quotes:
                alg::erase_tail(in_str,1);
                alg::erase_head(in_str,1);
            }
            outval=boost::any(in_str);
            // std::cerr << "***DEBUG: returning from validate(...std::string*...) ***" << std::endl;
        }
    } // program_options
} // boost
