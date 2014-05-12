/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2010 - 2011 by Lukas Gamper <gamperl@gmail.com>                   *
 *                                                                                 *
 * This software is part of the ALPS libraries, published under the ALPS           *
 * Library License; you can use, redistribute it and/or modify it under            *
 * the terms of the license, either version 1 or (at your option) any later        *
 * version.                                                                        *
 *                                                                                 *
 * You should have received a copy of the ALPS Library License along with          *
 * the ALPS Libraries; see the file LICENSE.txt. If not, the license is also       *
 * available from http://alps.comp-phys.org/.                                      *
 *                                                                                 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,        *
 * FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT       *
 * SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE       *
 * FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,     *
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER     *
 * DEALINGS IN THE SOFTWARE.                                                       *
 *                                                                                 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

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
