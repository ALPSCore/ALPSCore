/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_PARAMS_INCLUDED
#define ALPS_PARAMS_INCLUDED

#include "alps/config.hpp"
#include "alps/hdf5/archive.hpp"
// #include "alps/params/paramvalue.hpp"
// #include "alps/params/paramproxy.hpp"
// #include "alps/params/paramiterator.hpp"

#ifdef ALPS_HAVE_PYTHON_DEPRECATED
    #include "alps/ngs/boost_python.hpp"
    #include "boost/python/dict.hpp"
#endif

#include "boost/filesystem.hpp"

// Serialization headers:
#include "boost/serialization/map.hpp"
#include "boost/serialization/vector.hpp"
#include "boost/serialization/string.hpp"
#include "boost/serialization/variant.hpp"

#include "boost/program_options.hpp"
// #include "boost/optional.hpp"
#include "boost/any.hpp"
#include "boost/tokenizer.hpp"

#ifdef ALPS_HAVE_MPI
    namespace boost{ namespace mpi{ class communicator; } }
#endif

#include <map>
#include <vector>
#include <string>
#include <algorithm>

#include "alps/params/option_type.hpp"

namespace alps {

  namespace params_ns {
    
/** params: Parameter types.

    FIXME!!! The interface is redefined, must be reviewed!!!

    Interface:

    1. The objects are copyable and default-constructible.

    1.1. The constructor needs (argc, argv) with an optional HDF5 path, or an (HDF5) archive wih an optional path.
    
    2. The parameters are accessed this way:

        params p(argc,argv);
        // ...parameter definition...
        double t=p["Temp"];

        An undefined parameter cannot be accessed (throws exception).

    2.1. The parameters can also be assigned this way:
    
        double t=300;
        // ...parameter definition...
        p["Temp"]=t;

        Once assigned, parameter type cannot be changed.

    2.2. An attempt to read an undefined parameter results in exception.

    2.3. An attempt to read a parameter of a different type results in a silent type casting between the scalar types,
    and results in exception if any of the types (LHS or RHS) are vector types or strings.

    3. Allowed scalar types: double, int, bool, std::string (FIXME: specify more)

    3.1. A special type of parameter: "trigger"; these parameters can be given only in the command line,
         do not accept associated values, and are considered boolean and "true" if present, "false" if absent.
         E.g., a pre-defined parameter '--help' is a trigger parameter.

    4. Allowed vector types: std::vector<T> for any scalar type T except std::string type.

    5. Way to define the parameters that are expected to be read from a file or command line:

        p.description("The description for --help")
         .define<int>("L", 50, "optional int parameter L with default 50")
         .define<double>("T", "required double parameter T with no default")
         .define("continue", "trigger parameter with boolean value")
        ;

      NOTE: definition of both short and long variants of an option,
      while allowed by boost::program_options, is prohibited by this library.

    5.1. It is a responsibility of the caller to check for the "--help" option.
         A convenience method checks for the option and outputs the description of the options.

    5.2. A parameter assigned explicitly before its definition cannot be defined.

    6. List parameters of type T are defined as

        p.define< std::vector<T> >("name","description");

       and accessed as:

        x=p["name"].as< std::vector<T> >();

    List parameters cannot have a default value.
    Lists of strings are not supported (undefined behavior: may or may not work).

    7. The class CONTAINS a (mutable) std::map from parameters names
    to `option_type`, which is populated every time the file is
    parsed. The class also delegates some methods of std::map (FIXME: is it needed?)

    8. When constructed from (argc,argv), the options are read from the command line first, then from a
    parameter file in ini-file format. The name of the file must be
    given in the command line. The command-line options take priority
    over file options. The following specifications are devised:

    8.1. The parser and the option map are combined --- it makes a user's life easier.
    
    8.2. The options can be defined any time --- probably in the constructor of the class that uses them.

    8.3. Defining an option invalidates the object state, requesting re-parsing.

    8.4. Parsing occurs and the parameter map is populated at the first access to the parameters ("lazy parsing").

    8.5. Unknown (undeclared) options are ignored --- possibly setting a flag "unknown options are present" (FIXME).
    
    8.6. Options can NOT be redefined --- subclasses must come up with
         their own option names. The description (the help message) can be redefined.

    9. The ini-file format allows empty lines and comment lines, but not garbage lines.

    9.1. The list values in the ini file are comma/space separated.

    9.2. The boolean values can be 0|1, yes|no, true|false (case insensitive), as specified by boost::program_options. 

    9.3. The strings in the ini file are read according to the following rules:
       1) Leading and trailing spaces are stripped.
       2) A pair of surrounding double quotes is stripped, if present (to allow for leading/trailing spaces).
         
    10. The state of a parameter object can be saved to and loaded from
    an HDF5 archive.

    11. The state of a parameter object can be broadcast over an MPI
    communicator. (FIXME: not yet implemented)

    QUESTIONS:
    
*/

  

      /// Namespace hiding boring/awkward implementation details
      namespace detail {

          /// Service cast-via-validate function from a string to (presumably scalar) type T
          template <typename T>
          static T validate_cast(const std::string& sval)
          {
              using boost::program_options::validate;
              std::vector<std::string> sval_vec(1);
              sval_vec[0]=sval;
              boost::any outval;
              validate(outval, sval_vec, (T*)0, 0);
              return boost::any_cast<T>(outval);
          }

          /// Service function: output a vector
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
        
          // template <typename T>
          // void printout(std::ostream& os, const boost::any& val)
          // {
          //     os << boost::any_cast<T>(val);
          // }

          // template <>
          // void printout<std::string>(std::ostream& os, const boost::any& val)
          // {
          //     typedef std::string T;
          //     os << "\"" << boost::any_cast<T>(val) << "\"";
          // }


      } // detail


      class /*-ALPS_DECL-*/ params {
      public:
          // typedef params_ns::options_map_type options_map_type;
      private:
          // typedef boost::program_options::options_description options_description;
          // typedef void (option_type::*assign_fn_type)(const boost::any&);
          // typedef std::map<std::string, assign_fn_type> anycast_map_type;

          
          // typedef boost::program_options::variables_map variables_map;
          // typedef void (*printout_type)(std::ostream&, const boost::any&);
          // typedef std::map<std::string,printout_type> printout_map_type;

          /// True if there are no new define<>()-ed parameters since last parsing. Mutated by deferred parsing: certainly_parse()
          mutable bool is_valid_;
          /// Options (parameters). Mutated by deferred parsing: certainly_parse()
          mutable options_map_type optmap_; 
          
          /// Map (option names --> definition). Filled by define<T>() method.
          detail::description_map_type descr_map_;
          
          std::string helpmsg_;                 ///< Help message
          std::vector<std::string> argvec_;     ///< Command line arguments
          std::string infile_;                  ///< File name to read from (if not empty)
          std::string argv0_;                   ///< 0-th argument (program name)
          boost::optional<std::string> archname_; ///< Archive name (if restored from archive)

          /// Parses the parameter file, filling the option map, and using the provided options_description instance
          void certainly_parse(boost::program_options::options_description&) const;
            
          /// Parses the parameter file, filling the option map.
          void certainly_parse() const
          {
              boost::program_options::options_description odescr;
              certainly_parse(odescr);
          }
        
          /// Parses the parameters if not already parsed.
          void possibly_parse() const { if (!is_valid_) certainly_parse(); }

          /// Invalidates the option map
          void invalidate() { is_valid_=false; }

          /// Initialization code common for all constructors
          void init() {
              is_valid_=false;
              this->define("help", "Provides help message");
          }

          /// Function to check for validity/redefinition of an option (throws!)
          void check_validity(const std::string& optname) const;

      public:
            
          typedef options_map_type::iterator iterator;
          typedef options_map_type::const_iterator const_iterator;
          typedef options_map_type::value_type value_type;
          typedef options_map_type::mapped_type mapped_type;

          // Some more convenience typedefs (exception types)
          /// Exception type: mismatched parameter types
          typedef option_type::type_mismatch type_mismatch;
          /// Exception type: attempt to use uninitialized option
          typedef option_type::uninitialized_value uninitialized_value;

          /// Exception type: attempt to redefine a parameter
          struct double_definition : public option_type::exception_base {
              double_definition(const std::string& a_name, const std::string& a_what)
                  : option_type::exception_base(a_name, a_what) {}
          };
          
          /// Exception type: attempt to define explicitly assigned parameter
          struct extra_definition : public option_type::exception_base {
              extra_definition(const std::string& a_name, const std::string& a_what)
                  : option_type::exception_base(a_name, a_what) {}
          };
          
          /// Exception type: incorrect parameter name
          struct invalid_name : public option_type::exception_base {
              invalid_name(const std::string& a_name, const std::string& a_what)
                  : option_type::exception_base(a_name, a_what) {}
          };

          /// Exception type: the object was not restored from archive
          struct not_restored : public std::runtime_error {
              not_restored(const std::string& a_what)
                  : std::runtime_error(a_what) {}
          };
          
          /** Default constructor */
          params() { init(); }

          /** Constructor from HDF5 archive. */
          params(hdf5::archive ar, std::string const & path = "/parameters")
          {
              this->load(ar, path);
          }

          /// Constructor from command line and a parameter file. The parsing is deferred.
          /** Tries to see if the file is an HDF5, in which case restores the object from the
              HDF5 file, ignoring the command line.
              @param hdfpath : path to HDF5 dataset containing the saved parameter object
                               (NULL if this functionality is not needed)
          */
          params(unsigned int argc, const char* argv[], const char* hdfpath = "/parameters");

#ifdef ALPS_HAVE_PYTHON_DEPRECATED
          params(boost::python::dict const & arg);
          params(boost::python::str const & arg);
#endif

          /// Returns whether the parameters are restored from archive by the constructor
          bool is_restored() const
          {
              return bool(archname_);
          }

          /// Returns the name of the archive the parameters were restarted from (or throw)
          std::string get_archive_name() const
          {
              if (archname_) return *archname_;
              throw not_restored("This instance of parameters was not restored from an archive");
          }

          /// Returns the "base name": (parameter file name) || (restart file name) || (program name) || (empty string)
          std::string get_base_name() const
          {
              if (!infile_.empty()) return infile_;
              if (archname_) return *archname_;
              return argv0_;
          }
          
          /** Returns number of parameters (size of the map) */
          std::size_t size() const { possibly_parse(); return optmap_.size(); }

          /** Erase a parameter */
          void erase(std::string const& k) { possibly_parse(); optmap_.erase(k); }

          /** Check if the parameter is present. FIXME: semantics?? */
          bool defined(std::string const & key) const
          {
              throw std::logic_error("Called defined('"+key+"'): semantic is not yet clearly defined.");
              possibly_parse();
              return (optmap_.count(key)!=0);
          }

          /** Returns iterator to the beginning of the option map */
          iterator begin() { possibly_parse(); return optmap_.begin(); }
            
          /** Returns iterator to the beginning of the option map */
          const_iterator begin() const { possibly_parse(); return optmap_.begin(); }

          /** Iterator to the beyond-the-end of the option map */
          iterator end() { possibly_parse(); return optmap_.end(); }

          /** Iterator to the beyond-the-end of the option map */
          const_iterator end() const { possibly_parse(); return optmap_.end(); }

          /** Access a parameter: read-only */
          const mapped_type& operator[](const std::string& k) const;

          /** Access a parameter --- possibly for assignment */
          mapped_type& operator[](const std::string& k);

          /// Save parameters to HDF5 archive
          void save(hdf5::archive &) const;

          /// Save parameters to HDF5 archive to a given path
          void save(hdf5::archive &, const std::string&) const;

          /// Load parameters from HDF5 archive (overwriting the object)
          void load(hdf5::archive &);

          /// Load parameters from HDF5 archive (overwriting the object) from a given path
          void load(hdf5::archive &, const std::string&);

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

          /// Define an option of type T with an optional value
          template <typename T>
          params& define(const std::string& optname, T defval, const std::string& descr);

          /// Define an option of type T with a required value
          template <typename T>
          params& define(const std::string& optname, const std::string& descr);

          /// Define a "trigger" command-line option (like "--help")
          params& define(const std::string& optname, const std::string& a_descr);

          /// Output the help message, if requested. @returns true if help was indeed requested.
          bool help_requested(std::ostream& ostrm);

      private:

          friend class boost::serialization::access;

          /// Interface to serialization
          template<class Archive> void serialize(Archive & ar, const unsigned int)
          {
              ar  & is_valid_
                  & archname_
                  & optmap_
                  & descr_map_
                  & helpmsg_
                  & argvec_
                  & infile_
                  & argv0_;
          }
      };

      // FIXME: we may consider provide template specializations for specific types? To hide templates inside *.cpp?

      /// Define an option of a generic type with a default value
      template <typename T>
      inline params& params::define(const std::string& optname, T defval, const std::string& a_descr)
      {
          check_validity(optname);
          invalidate();
          typedef detail::description_map_type::value_type value_type;
          bool result=descr_map_.insert(value_type(optname, detail::option_description_type(a_descr,defval))).second;
          assert(result && "The inserted element is always new");
          return *this;
      }

      /// Define an option of a generic type without default
      template <typename T>
      inline params& params::define(const std::string& optname, const std::string& a_descr)
      {
          check_validity(optname);
          invalidate();
          typedef detail::description_map_type::value_type value_type;
          bool result=descr_map_.insert(value_type(optname, detail::option_description_type(a_descr, (T*)0))).second;
          assert(result && "The inserted element is always new");
          return *this;
      }

      /// Define a "trigger" option
      inline params& params::define(const std::string& optname, const std::string& a_descr)
      {
          check_validity(optname);
          invalidate();
          typedef detail::description_map_type::value_type value_type;
          bool result=descr_map_.insert(value_type(optname, detail::option_description_type(a_descr))).second;
          assert(result && "The inserted element is always new");
          return *this;
      }

      // /*-ALPS_DECL-*/ std::ostream & operator<<(std::ostream & os, params const & arg);

      namespace detail {
          /// Validator for vectors, used by boost::program_options
          template <typename T>
          void validate(boost::any& outval, const std::vector<std::string>& strvalues,
                        vector_tag<T>*, int)
          {
              namespace po=boost::program_options;
              namespace pov=po::validators;
              typedef std::vector<std::string> strvec;
              typedef boost::char_separator<char> charsep;
          
              // std::cerr << "***DEBUG: entering validate() (templated) ***" << std::endl;

              pov::check_first_occurrence(outval); // check that this option has not yet been assigned
              const std::string in_str=pov::get_single_string(strvalues); // check that this option is passed a single value

              // Now, do parsing
              boost::tokenizer<charsep> tok(in_str,charsep(" ;,"));
              const strvec tokens(tok.begin(),tok.end());
              std::vector<T> typed_outval(tokens.size());
              std::transform(tokens.begin(), tokens.end(), typed_outval.begin(), detail::validate_cast<T>);

              outval=boost::any(typed_outval);
              // std::cerr << "***DEBUG: returning from validate() (templated) ***" << std::endl;
          }
      } // detail
  } // params_ns

    // Elevate relevant class to the alps NS:
    using params_ns::params;
    
} // alps

namespace boost {
    namespace program_options {
        /// Validator for std::string, overriding one defined by boost::program_options
        // It has to be declared in boost::program_options namespace to work!
        void validate(boost::any& outval, const std::vector<std::string>& strvalues,
                      std::string*, int);
    }
}

#endif // ALPS_PARAMS_INCLUDED
