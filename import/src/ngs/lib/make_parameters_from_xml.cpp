/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/ngs/make_deprecated_parameters.hpp>
#include <alps/ngs/make_parameters_from_xml.hpp>

#include <string>

namespace alps {

    ALPS_DECL params make_parameters_from_xml(boost::filesystem::path const & arg) {
        Parameters par;
        boost::filesystem::ifstream infile(arg.string());

        // read outermost tag (e.g. <SIMULATION>)
        XMLTag tag = parse_tag(infile, true);
        std::string closingtag = "/" + tag.name;

        // scan for <PARAMETERS> and read them
        tag = parse_tag(infile, true);
        while (tag.name != "PARAMETERS" && tag.name != closingtag) {
            std::cerr << "skipping tag with name " << tag.name << "\n";
            skip_element(infile, tag);
            tag = parse_tag(infile, true);
        }

        par.read_xml(tag, infile, true);
        if (!par.defined("SEED"))
            par["SEED"] = 0;
        
        params res;
        for (Parameters::const_iterator it = par.begin(); it != par.end(); ++it)
            res[it->key()] = it->value();
        return res;
    }
}
