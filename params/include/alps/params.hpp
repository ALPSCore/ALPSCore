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

      } // detail


      class /*-ALPS_DECL-*/ params {
      private:

          /// Type for the set of key names. Used in recording defaulted options. (FIXME: think of a better solution)
          typedef std::set<std::string> keys_set_type;
          
          /// True if there are no new define<>()-ed parameters since last parsing. Mutated by deferred parsing: certainly_parse()
          mutable bool is_valid_;
          /// Options (parameters). Mutated by deferred parsing: certainly_parse()
          mutable options_map_type optmap_; 
          
          /// Set of options having their default value. Mutated by deferred parsing: certainly_parse()
          // FIXME: it is a quick hack. Should be merged with descr_map_ below.
          mutable keys_set_type defaulted_options_;
          
          /// Map (option names --> definition). Filled by define<T>() method.
          detail::description_map_type descr_map_;

          std::string helpmsg_;                 ///< Help message
          std::vector<std::string> argvec_;     ///< Command line arguments
          std::string infile_;                  ///< File name to read from (if not empty)
          std::string argv0_;                   ///< 0-th argument (program name)
          boost::optional<std::string> archname_; ///< Archive name (if restored from archive)

          /// Parses the parameter file, filling the option map, and using the provided options_description instance
          void certainly_parse(boost::program_options::options_description&, bool reassign=false) const;
            
          /// Parses the parameter file, filling the option map.
          void certainly_parse(bool reassign=false) const
          {
              boost::program_options::options_description odescr;
              certainly_parse(odescr,reassign);
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

          /// @brief Convenience function: returns the "origin name"
          /// @Returns (parameter_file_name || restart_file name || program_name || "")
          std::string get_origin_name() const;
        
          /** Returns number of parameters (size of the map) */
          std::size_t size() const { possibly_parse(); return optmap_.size(); }

          /** Access a parameter: read-only */
          const mapped_type& operator[](const std::string& k) const
          {
              possibly_parse();
              return optmap_[k];
          }

          /** Access a parameter: possibly for assignment */
          mapped_type& operator[](const std::string& k)
          {
              possibly_parse();
              return optmap_[k];
          }

          /** Returns iterator to the beginning of the option map */
          const_iterator begin() const { possibly_parse(); return optmap_.begin(); }

          /** Iterator to the beyond-the-end of the option map */
          const_iterator end() const { possibly_parse(); return optmap_.end(); }

          /// Check if a parameter is defined (that is: attempt to assign to/from it will not throw)
          bool defined(const std::string& name) const
          {
              possibly_parse();
              options_map_type::const_iterator it=optmap_.find(name);
              return (it!=optmap_.end()) && !(it->second).isNone();
          }

          /// Check if parameter has default value (does not present in the command line).
          /// @remark For non-existing and implicitly defined (by assignment) parameters the result is undefined (currently: false).
          // FIXME: what if it does not exist? what if it was explicitly assigned?
          bool defaulted(const std::string& name) const
          {
              possibly_parse();
              return defaulted_options_.count(name)!=0; // FIXME: the implementation via set is a quick hack
          }
        

          /// Save parameters to HDF5 archive
          void save(hdf5::archive &) const;

          /// Save parameters to HDF5 archive to a given path
          void save(hdf5::archive &, const std::string&) const;

          /// Load parameters from HDF5 archive (overwriting the object)
          void load(hdf5::archive &);

          /// Load parameters from HDF5 archive (overwriting the object) from a given path
          void load(hdf5::archive &, const std::string&);

#ifdef ALPS_HAVE_MPI
          /// Broadcast the parameters to all processes (FIXME: does not have a test yet)
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

          /// Stream parameters 
          friend std::ostream& operator<<(std::ostream& str, params const& x);

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
#ifndef NDEBUG
          bool result= 
#endif
          descr_map_.insert(value_type(optname, detail::option_description_type(a_descr,defval)))
#ifndef NDEBUG
          .second;
          assert(result && "The inserted element is always new");
#else
          ;
#endif
          return *this;
      }

      /// Define an option of a generic type without default
      template <typename T>
      inline params& params::define(const std::string& optname, const std::string& a_descr)
      {
          check_validity(optname);
          invalidate();
          typedef detail::description_map_type::value_type value_type;
#ifndef NDEBUG
          bool result=
#endif
          descr_map_.insert(value_type(optname, detail::option_description_type(a_descr, (T*)0)))
#ifndef NDEBUG
          .second;
          assert(result && "The inserted element is always new");
#else
          ;
#endif
          return *this;
      }

      /// Define a "trigger" option
      inline params& params::define(const std::string& optname, const std::string& a_descr)
      {
          check_validity(optname);
          invalidate();
          typedef detail::description_map_type::value_type value_type;
#ifndef NDEBUG
          bool result=
#endif
          descr_map_.insert(value_type(optname, detail::option_description_type(a_descr)))
#ifndef NDEBUG
          .second;
          assert(result && "The inserted element is always new");
#else
          ;
#endif
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
