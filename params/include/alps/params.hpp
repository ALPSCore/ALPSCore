/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#pragma once

#include "alps/config.hpp"
#include "alps/hdf5/archive.hpp"
#include "alps/params/paramvalue.hpp"
#include "alps/params/paramproxy.hpp"
#include "alps/params/paramiterator.hpp"

#ifdef ALPS_HAVE_PYTHON_DEPRECATED
    #include "alps/ngs/boost_python.hpp"
    #include "boost/python/dict.hpp"
#endif

#include "boost/filesystem.hpp"
#include "boost/serialization/map.hpp"
#include "boost/serialization/vector.hpp"
#include "boost/serialization/string.hpp" 
#include "boost/program_options.hpp"
#include "boost/optional.hpp"
#include "boost/any.hpp"

#ifdef ALPS_HAVE_MPI
    namespace boost{ namespace mpi{ class communicator; } }
#endif

#include <map>
#include <vector>
#include <string>
#include <algorithm>

namespace alps {

/** params: Parameter types.

    Interface:

    1. The objects are copyable and default-constructible.

    1.1. The constructor needs (argc, argv), a file name, or an (HDF5) archive
    
    2. The parameters are accessed this way:

        params p(argc,argv);
        double t=p["Temp"].as<double>();

    3. Allowed scalar types: double, int, std::string, any type T for
    which boost::lexical_cast<T>() is defined.
    
    4. Allowed vector types: alps::params::intvec,
    alps::params::doublevec for which a parser is defined.

    Variant: use alps::param::vector<T> which is derived from
    std::vector<T> and can parse a comma or space-separated list of
    values of type T.

    5. Way to define the parameters:

        p.description("The description for --help")
         .define<int>("L", 50, "optional int parameter L with default 50")
         .define<double>("T", "required double parameter T with no default")
        ;

    5.1. It is a responsibility of the caller to check for the "--help" option.
         A convenience method checks for the option and outputs the description of the options.

    6. Special case: parameter with no declared type is optional,
       corresponds to a boolean TRUE if it is defined and true:

        p.define<>("X", "The input has property X");
        bool has_x=p["X"].as<bool>();

    7. The type is NOT derived from, but it CONTAINS (mutable) boost::program_options::variables_map;
    it delegates some methods of std::map.

    8. When constructed from (argc,argv), The options are read from the command line first, then from a
    parameter file in ini-file format. The name of the file must be
    given in the command line. The command-line options take priority
    over file options. The following specifications are devised:

    8.1. The parser and the option map are combined --- it makes a user's life easier.
    
    8.2. The options can be defined any time --- probably in the constructor of the class that uses them.

    8.3. Defining an option invalidates (clears) the option map.

    8.4. Parsing occurs and the option map populated at the first access to the option map ("lazy parsing").

    8.5. Unknown (undeclared) options are ignored --- possibly setting a flag "unknown options are present".
    
    8.6. Options can NOT be redefined --- subclasses must come up with
         their own option names. The description (the help message) can be redefined.
    
    9. The state of a parameter object can be saved to and loaded from
    an HDF5 archive.

    10. The state of a parameter object can be broadcast over an MPI
    communicator.
    
*/
    class /*-ALPS_DECL-*/ params {


        // typedef std::map<std::string, detail::paramvalue>::value_type iterator_value_type;

        // friend class detail::paramiterator<params, iterator_value_type>;
        // friend class detail::paramiterator<params const, iterator_value_type const>;

            typedef boost::program_options::variables_map variables_map;
            typedef boost::program_options::options_description options_description;
            typedef void (*printout_type)(std::ostream&);

            /// Contains map(options->values), or empty until a file is parsed. Mutated by diferred fiel parsing.
            mutable boost::optional<variables_map> varmap_;

            /// Map(options->output_functions); filled by define() method or direct assignment
            std::map<std::string,printout_type> printout_map_;

            /// Options description; filled by define() method
            options_description descr_;

            std::string helpmsg_;
            std::vector<std::string> argvec_;
            //?? std::string file_;

            /// An option name for the positional file argument.
            static const char* const cfgfile_optname_;

            /// Parses the parameter file, filling the option map.
            void certainly_parse() const;
            
            /// Parses the parameters if not already parsed.
            void possibly_parse() const { if (!varmap_) certainly_parse(); }

            /// Invalidates the option map
            void invalidate() { varmap_=boost::none; }

            /// Initialization code common for all constructors
            void init() {
                descr_.add_options()
                    ("parameter-file",boost::program_options::value<std::string>())
                    ("help","Provides help message");
            }

            /// Service functor class to convert C-string pointer to an std::string
            struct cstr2string {
                    std::string operator()(const char* cstr)
                    {
                        return std::string(cstr);
                    }
            };
            
        public:

            typedef variables_map::iterator iterator;
            typedef variables_map::const_iterator const_iterator;
            typedef variables_map::value_type value_type;
            typedef variables_map::mapped_type mapped_type;
            // typedef variables_map::value_type value_type;

            // old: typedef detail::paramiterator<params, iterator_value_type> iterator;
            // old: typedef detail::paramiterator<params const, iterator_value_type const> const_iterator;
            // old: typedef detail::paramproxy value_type;

        
            /** Default constructor */
            params() { init(); }

            // /** Copy constructor */
            // params(params const & arg): {}

            /** Constructor from HDF5 archive. (FIXME: not implemented yet) */
            params(hdf5::archive ar, std::string const & path = "/parameters");

            // /** Constructor from parameter file. The parsing of the file is deferred. */
            // params(boost::filesystem::path const &);

            /// Constructor from command line and a parameter file. The parsing is deferred. 
            params(unsigned int argc, const char* const argv[])
            {
                std::transform(argv+1,argv+argc, std::back_inserter(argvec_), cstr2string());
            }

            #ifdef ALPS_HAVE_PYTHON_DEPRECATED
                params(boost::python::dict const & arg);
                params(boost::python::str const & arg);
            #endif

            /** Returns number of parameters (size of the map) */
            std::size_t size() const { possibly_parse(); return varmap_->size(); }

            /** Erase a parameter */
            void erase(std::string const& k) { possibly_parse(); varmap_->erase(k); }

            // /** Access a parameter */
            // boost::any& operator[](const std::string& k)
            // {
            //     possibly_parse();
            //     std::map<std::string,mapped_type>& as_map=*varmap_; //FIXME? does it work as expected?
            //     // return (*varmap_)[k].value();
            //     return as_map[k].value();
            // }

            /** Access a parameter: read-only */
            const mapped_type& operator[](const std::string& k) const
            {
                possibly_parse();
                return (*varmap_)[k];
            }

            /** Check if the parameter is defined */
            bool defined(std::string const & key) const
            {
                possibly_parse();
                return (varmap_->count(key)!=0);
            }

            /** Returns iterator to the beginning of the option map */
            iterator begin() { possibly_parse(); return varmap_->begin(); }
            
            /** Returns iterator to the beginning of the option map */
            const_iterator begin() const { possibly_parse(); return varmap_->begin(); }

            /** Iterator to the beyond-the-end of the option map */
            iterator end() { possibly_parse(); return varmap_->end(); }

            /** Iterator to the beyond-the-end of the option map */
            const_iterator end() const { possibly_parse(); return varmap_->end(); }
            
            /// Save parameters to HDF5 archive (FIXME: not implemented yet)
            void save(hdf5::archive &) const;

            /// Load parameters from HDF5 archive (clearing the object first) (FIXME: not implemented yet)
            void load(hdf5::archive &);

            #ifdef ALPS_HAVE_MPI
            /// Broadcast the parameters to all processes (FIXME: not implemented yet)
                void broadcast(boost::mpi::communicator const &, int = 0);
            #endif

            // -- now for the defining the options ---

            /// Set the help text for '--help' option
            params& description(const std::string& helpline)
            {
                invalidate();
                helpmsg_=helpline;
                return *this;
            }

            /// Define an option with an optional value
            template <typename T>
            params& define(const std::string& optname, T defval, const std::string& descr);

            /// Define an option with a required value
            template <typename T>
            params& define(const std::string& optname, const std::string& descr);

            /// Output the help message, if requested. @returns true if help was indeed requested.
            bool help_requested(std::ostream& ostrm);

            // FIXME: what the following should mean exactly?
            //        possibly a shortcut for a boolean option with default false?
            // /// Define an option that may be omitted from the parameter file
            // template <>
            // params& define<>(const std::string& optname, const std::string& descr);

        private:

            friend class boost::serialization::access;

            // FIXME: implement serialization
            // template<class Archive> void serialize(Archive & ar, const unsigned int) {
            //     ar & keys
            //        & values
            //     ;
            // }

            // void setter(std::string const &, detail::paramvalue const &);
            // void parse_text_parameters(boost::filesystem::path const & path);

            // detail::paramvalue getter(std::string const &);

            // std::vector<std::string> keys;
            // std::map<std::string, detail::paramvalue> values;
    };

    namespace detail {

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
        
        template <typename T>
        void printout(std::ostream& os, const boost::any& val)
        {
          os << boost::any_cast<T>(val);
        }

        template <>
        void printout<std::string>(std::ostream& os, const boost::any& val)
        {
          typedef std::string T;
          os << "'" << boost::any_cast<T>(val) << "'";
        }
    }

    // FIXME: we may consider provide template specializations for specific types? To hide templates inside *.cpp?
    template <typename T>
    params& params::define(const std::string& optname, T defval, const std::string& a_descr)
    {
        invalidate();
        descr_.add_options()(optname,boost::program_options::value<T>()->default_value(defval),a_descr);
        printout_map_[optname]=detail::printout<T>; 
        return *this;
    }

    template <typename T>
    params& params::define(const std::string& optname, const std::string& a_descr)
    {
        invalidate();
        descr_.add_options()(optname,boost::program_options::value<T>(),a_descr);
        return *this;
    }

    /*-ALPS_DECL-*/ std::ostream & operator<<(std::ostream & os, params const & arg);

    /// Assign a value to a parameter, as in param["abc"] << x;
    template <typename T>
    /*-ALPS_DECL-*/ void operator<<(params::mapped_type& slot, const T& val)
    {
        possibly_parse();
        std::map<std::string,params::mapped_type>& as_map=*varmap_; //FIXME? does it work as expected?
        as_map[k].value()=val;
        printout_map_[k]=detail::printout<T>; 
    }
}
