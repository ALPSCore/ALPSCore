/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2011 - 2013 by Lukas Gamper <gamperl@gmail.com>                   *
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

#define BOOST_TEST_MODULE alps::ngs::accumulator

#include <alps/hdf5/array.hpp>
#include <alps/hdf5/multi_array.hpp>

#include <alps/ngs/random01.hpp>
#include <alps/ngs/accumulator/accumulator.hpp>

#include <alps/alea/mcdata.hpp>
#include <alps/alea/observableset.h>
#include <alps/alea/detailedbinning.h>

#ifndef ALPS_LINK_BOOST_TEST
#	include <boost/test/included/unit_test.hpp>
#else
#	include <boost/test/unit_test.hpp>
#endif

alps::alea::mcdata<double> make_result(alps::Observable const & obs) {
    return alps::alea::mcdata<double>(dynamic_cast<alps::AbstractSimpleObservable<double> const &>(obs));
}

BOOST_AUTO_TEST_CASE(ngs_alea_compare) {
	using alps::accumulator::accumulator_set;
	using alps::accumulator::result_set;
	using alps::ObservableSet;
	using alps::alea::mcdata;

	accumulator_set accumulators;
	ObservableSet observables;
	alps::random01 random(42);

	accumulators 
		<< alps::accumulator::RealObservable("Scalar")
		<< alps::accumulator::RealObservable("Correlated")
	;
	observables 
		<< alps::RealObservable("Scalar")
		<< alps::RealObservable("Correlated")
	;

	BOOST_REQUIRE(count(accumulators["Scalar"]) == 0);

	accumulators["Scalar"] << 1.;
	observables["Scalar"] << 1.;

	{
		result_set results(accumulators);
		mcdata<double> scalar_result = make_result(observables["Scalar"]);

		BOOST_REQUIRE(count(accumulators["Scalar"]) == 1);
		BOOST_REQUIRE(count(results["Scalar"]) == 1);
		BOOST_REQUIRE(scalar_result.count() == 1);
	}

	double corr = random();
	for (int i = 0; i < 1000000; ++i) {
		double rng = random();
		corr = (corr + rng) / 2;
		accumulators["Scalar"] << rng;
		observables["Scalar"] << rng;
		accumulators["Correlated"] << corr;
		observables["Correlated"] << corr;
	}

	{
		std::cout << "accumulator" << std::endl;

		std::cout << "mean  new: " << mean(accumulators["Scalar"].get<double>()) << "\told: " << dynamic_cast<alps::RealObservable const &>(observables["Scalar"]).mean() << "\tdiff: "
			<< mean(accumulators["Scalar"].get<double>()) - dynamic_cast<alps::RealObservable const &>(observables["Scalar"]).mean() << std::endl;
		std::cout << "error new: " << error(accumulators["Scalar"].get<double>()) << "\told: " << dynamic_cast<alps::RealObservable const &>(observables["Scalar"]).error() << "\tdiff: "
			<< error(accumulators["Scalar"].get<double>()) - dynamic_cast<alps::RealObservable const &>(observables["Scalar"]).error() << std::endl;

		std::cout << "correlated new: " << accumulators["Correlated"] << std::endl;
		std::cout << "correlated old: " << observables["Correlated"] << std::endl;
	}

	{
		result_set results(accumulators);
		mcdata<double> scalar_result = make_result(observables["Scalar"]);
		mcdata<double> correlated_result = make_result(observables["Correlated"]);

		BOOST_REQUIRE(count(accumulators["Scalar"]) == 1000001);
		BOOST_REQUIRE(count(results["Scalar"]) == 1000001);
		BOOST_REQUIRE(scalar_result.count() == 1000001);

		std::cout << std::endl << "result" << std::endl;

		std::cout << "uncorrelated mean  new: " << mean(results["Scalar"].get<double>()) << "\told: " << scalar_result.mean() << "\tdiff: "
			<< results["Scalar"].mean<double>() - scalar_result.mean() << std::endl;
		std::cout << "uncorrelated error new: " << error(results["Scalar"].get<double>()) << "\told: " << scalar_result.error() << "\tdiff: "
			<< results["Scalar"].error<double>() - scalar_result.error() << std::endl;
		std::cout << "correlated mean  new: " << mean(results["Correlated"].get<double>()) << "\told: " << correlated_result.mean() << "\tdiff: "
			<< results["Correlated"].mean<double>() - correlated_result.mean() << std::endl;
		std::cout << "correlated error new: " << error(results["Correlated"].get<double>()) << "\told: " << correlated_result.error() << "\tdiff: "
			<< results["Correlated"].error<double>() - correlated_result.error() << std::endl;

		BOOST_REQUIRE_SMALL((error(results["Scalar"].get<double>()) - scalar_result.error()) / scalar_result.mean(), 1e-3);
		BOOST_REQUIRE_SMALL((mean(results["Scalar"].get<double>()) - scalar_result.mean()) / scalar_result.mean(), 1e-3);

		std::cout << std::endl << "result * result" << std::endl;

		result_set::value_type transformed_scalar_new = results["Scalar"] * results["Scalar"];
		mcdata<double> transformed_scalar_old = scalar_result * scalar_result;

		std::cout << "uncorrelated mean  new: " << mean(transformed_scalar_new.get<double>()) << "\told: " << transformed_scalar_old.mean() << "\tdiff: "
			<< (mean(transformed_scalar_new.get<double>()) - transformed_scalar_old.mean()) << std::endl;
		std::cout << "uncorrelated error new: " << error(transformed_scalar_new.get<double>()) << "\told: " << transformed_scalar_old.error() << "\tdiff: "
			<< (error(transformed_scalar_new.get<double>()) - transformed_scalar_old.error()) << std::endl;

		BOOST_REQUIRE_SMALL((mean(transformed_scalar_new.get<double>()) - transformed_scalar_old.mean()) / transformed_scalar_old.mean(), 1e-3);
		BOOST_REQUIRE_SMALL((error(transformed_scalar_new.get<double>()) - transformed_scalar_old.error()) / transformed_scalar_old.error(), 1e-3);

		result_set::value_type transformed_correlated_new = results["Correlated"] * results["Correlated"];
		mcdata<double> transformed_correlated_old = correlated_result * correlated_result;

		std::cout << "correlated mean  new: " << mean(transformed_correlated_new.get<double>()) << "\told: " << transformed_correlated_old.mean() << "\tdiff: "
			<< (mean(transformed_correlated_new.get<double>()) - transformed_correlated_old.mean()) << std::endl;
		std::cout << "correlated error new: " << error(transformed_correlated_new.get<double>()) << "\told: " << transformed_correlated_old.error() << "\tdiff: "
			<< (error(transformed_correlated_new.get<double>()) - transformed_correlated_old.error()) << std::endl;

		{
			result_set::value_type sin_scalar_new = results["Correlated"].transform<double>((double(*)(double))&std::sin);
			std::cout << "sin(correlated mean): " << sin_scalar_new.mean<double>() << std::endl;
		}

		{
			boost::function<double(double)> fkt_p((double(*)(double))&std::sin);
			result_set::value_type sin_scalar_new = results["Correlated"].transform(fkt_p);
			mcdata<double> sin_scalar_old = sin(correlated_result);

			std::cout << "sin(correlated mean)  new: " << sin_scalar_new.mean<double>() << "\told: " << sin_scalar_old.mean() << "\tdiff: "
				<< (sin_scalar_new.mean<double>() - sin_scalar_old.mean()) << std::endl;
			std::cout << "sin(correlated error)  new: " << sin_scalar_new.error<double>() << "\told: " << sin_scalar_old.error() << "\tdiff: "
				<< (sin_scalar_new.error<double>() - sin_scalar_old.error()) << std::endl;

			BOOST_REQUIRE_SMALL((sin_scalar_new.mean<double>() - sin_scalar_old.mean()) / sin_scalar_old.mean(), 1e-3);
			BOOST_REQUIRE_SMALL((sin_scalar_new.error<double>() - sin_scalar_old.error()) / sin_scalar_old.error(), 1e-3);
		}

		{
			boost::function<double(double, double)> fkt_p = alps::numeric::plus<double, double, double>();
			alps::accumulator::RealObservable::result_type add_scalar_new_acc = results["Correlated"].extract<alps::accumulator::RealObservable::result_type>();
			add_scalar_new_acc.transform(fkt_p, add_scalar_new_acc);
			result_set::value_type add_scalar_new(add_scalar_new_acc);

			mcdata<double> add_scalar_old = correlated_result + correlated_result;

			std::cout << "2 * (correlated mean)  new: " << add_scalar_new.mean<double>() << "\told: " << add_scalar_old.mean() << "\tdiff: "
				<< (add_scalar_new.mean<double>() - add_scalar_old.mean()) << std::endl;

			std::cout << "2 * (correlated error)  new: " << add_scalar_new.error<double>() << "\told: " << add_scalar_old.error() << "\tdiff: "
				<< (add_scalar_new.error<double>() - add_scalar_old.error()) << std::endl;

			BOOST_REQUIRE_SMALL((add_scalar_new.mean<double>() - add_scalar_old.mean()) / add_scalar_old.mean(), 1e-3);
			BOOST_REQUIRE_SMALL((add_scalar_new.error<double>() - add_scalar_old.error()) / add_scalar_old.error(), 1e-3);
		}
	}

}
