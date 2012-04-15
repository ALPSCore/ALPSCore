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

#include <alps/ngs/params.hpp>

#include <alps/parameter.h>

#include <boost/bind.hpp>

#include <algorithm>

namespace alps {

	params::params(hdf5::archive ar, std::string const & path) {
		std::string context = ar.get_context();
		ar.set_context(path);
		load(ar);
		ar.set_context(context);
	}

	params::params(boost::filesystem::path const & path) {
		boost::filesystem::ifstream ifs(path);
		Parameters par(ifs);
		for (Parameters::const_iterator it = par.begin(); it != par.end(); ++it) {
			detail::paramvalue val(it->value());
			setter(it->key(), val);
		}
	}

	#ifdef ALPS_HAVE_PYTHON
		params::params(boost::python::dict const & arg) {
			boost::python::extract<boost::python::dict> dict(arg);
			if (!dict.check())
				throw std::invalid_argument("parameters can only be created from a dict" + ALPS_STACKTRACE);
			const boost::python::object kit = dict().iterkeys();
			const boost::python::object vit = dict().itervalues();
			for (std::size_t i = 0; i < boost::python::len(dict()); ++i)
				setter(boost::python::call_method<std::string>(kit.attr("next")().ptr(), "__str__"), vit.attr("next")());
		}
	#endif

	std::size_t params::size() const {
		return keys.size();
	}

	void params::erase(std::string const & key) {
		if (!defined(key))
			throw std::invalid_argument("the key " + key + " does not exists" + ALPS_STACKTRACE);
		keys.erase(find(keys.begin(), keys.end(), key));
		values.erase(key);
	}

	params::value_type params::operator[](std::string const & key) {
		return value_type(
			defined(key),
			boost::bind(&params::getter, boost::ref(*this), key),
			boost::bind(&params::setter, boost::ref(*this), key, _1)
		);
	}

	params::value_type const params::operator[](std::string const & key) const {
		return defined(key)
			? value_type(values.find(key)->second)
			: value_type()
		;
	}

	bool params::defined(std::string const & key) const {
		return values.find(key) != values.end();
	}

	params::iterator params::begin() {
		return iterator(*this, keys.begin());
	}

	params::const_iterator params::begin() const {
		return const_iterator(*this, keys.begin());
	}

	params::iterator params::end() {
		return iterator(*this, keys.end());
	}

	params::const_iterator params::end() const {
		return const_iterator(*this, keys.end());
	}

	void params::save(hdf5::archive & ar) const {
		for (params::const_iterator it = begin(); it != end(); ++it)
			ar << make_pvp(it->first, it->second);
	}

	void params::load(hdf5::archive & ar) {
		keys.clear();
		values.clear();
		std::vector<std::string> list = ar.list_children(ar.get_context());
		for (std::vector<std::string>::const_iterator it = list.begin(); it != list.end(); ++it) {
			detail::paramvalue value;
			ar >> make_pvp(*it, value);
			setter(*it, value);
		}
	}

	#ifdef ALPS_HAVE_MPI
		void params::broadcast(boost::mpi::communicator const & comm, int root) {
			boost::mpi::broadcast(comm, *this, root);
		}
	#endif
	
	void params::setter(std::string const & key, detail::paramvalue const & value) {
		if (!defined(key))
			keys.push_back(key);
		values[key] = value;
	}

	detail::paramvalue params::getter(std::string const & key) {
		return values[key];
	}
}
