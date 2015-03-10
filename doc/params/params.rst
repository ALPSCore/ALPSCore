alps::params: Parameter types.
==============================

Interface:
----------

#. The objects are copyable and default-constructible.

#. The constructor needs ``(argc, argv)`` with an optional HDF5 path, or an (HDF5) archive with an optional path.

#. The parameters object can be constructed from an HDF5 archive or an INI
   file given as the first argument in the command line; the rest of the command line is then
   processed, overriding the options restored from the archive or read
   from the file.

#. The parameters are accessed this way: ::

        params p(argc,argv);
        // ...parameter definition...
        double t=p["Temp"];  // implicit type cast
        double t2=p["Temp"].as<double>()+123.4; // explicit type cast

   An undefined parameter cannot be accessed (throws exception).

#. The parameters can also be assigned this way: ::

        double t=300;
        p["Temp"]=t; // creates a parameter of type double

   Once assigned, parameter type cannot be changed.

#. An attempt to read a parameter of a different type results in a silent type casting between the scalar types,
   and results in exception if any of the types (LHS or RHS) are vector types or strings.

#. Allowed scalar types: ``double``, ``int``, ``bool``, ``std::string``

#. A special type of parameter: "trigger"; these parameters can be given only in the command line,
   do not accept associated values, and are considered boolean and ``true`` if present, ``false`` if absent.
   E.g., a pre-defined parameter ``--help`` is a trigger parameter.

#. Allowed vector types: ``std::vector<T>`` for any allowed scalar type ``T`` except ``std::string``.

#. Way to define the parameters that are expected to be read from a
   file or command line: ::

        p.description("The description for --help")
         .define<int>("L", 50, "optional int parameter L with default 50")
         .define<double>("T", "required double parameter T with no default")
         .define("continue", "trigger parameter with boolean value")
        ;

   *NOTE*: definition of both short and long variants of a parameter name,
   while allowed by ``boost::program_options``, is prohibited by this library.

#. It is a responsibility of the caller to check for the ``--help`` flag.
   A convenience method checks for the option and outputs the description of the options.

#. A parameter that has been assigned explicitly before its definition cannot be defined.

#. List parameters of type T are defined as ::

        p.define< std::vector<T> >("name","description");

   and accessed as: ::

        std::vector<T> x=p["name"];                          // implicit type cast
        size_t len=p["name"].as< std::vector<T> >().size();  //
        explicit type cast

   List parameters cannot have a default value.
   Lists of strings are not supported (undefined behavior: may or may not work).

#. Parameters can *NOT* be redefined. That means that subclasses that rely
   on their base class's parameter definitions must come up with
   their own option names. The description (the help message) can be
   redefined.

#. Unknown (undeclared) parameters both in command line and in INI file are ignored --- (*FIXME* may set a
   flag "unknown options are present" in a future version).
             
#. The ini-file format allows empty lines and comment lines, but not garbage lines.

#. The list values in the ini file are comma/space separated.

#. The boolean values can be ``0|1``, ``yes|no``, ``true|false`` (case insensitive), as specified by ``boost::program_options``. 

#. The strings in the ini file are read according to the following rules:
   1) Leading and trailing spaces are stripped.
   2) A pair of surrounding double quotes is stripped, if present (to allow for leading/trailing spaces).
           
#. The state of a parameter object can be saved to and loaded from
   an HDF5 archive.

#. The state of a parameter object can be broadcast over an MPI
   communicator.


Implementation notes
----------------------

#. The class CONTAINS a (mutable) ``std::map`` from parameters names
   to ``option_type``, which is populated every time the file is
   parsed. The class also delegates some methods of ``std::map``
   (*may be removed in the future*)

#. When constructed from ``(argc,argv)``, the options are read from the command line first, then from a
   parameter file in ini-file format. The name of the file must be
   given in the command line. The command-line options take priority
   over file options. The following specifications are devised:

        #) The parser and the option map are combined --- it makes a user's life easier.
                          
        #) The parameters can be defined any time --- probably in the constructor
           of the class that uses them. *However*, see a limitation below.

        #) Defining an option invalidates the object state, requesting
           re-parsing.

        #) *Therefore*, currently an attempt to
           define a parameter after loading the parameters object from an
           archive or receiving it by MPI broadcast results in undefined
           behavior (in particular, because the INI file may not be available).

        #) Parsing occurs and the parameter map is populated at the first access to the parameters ("lazy parsing").

