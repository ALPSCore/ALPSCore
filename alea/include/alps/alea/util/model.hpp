/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#pragma once

#include <alps/alea/core.hpp>
#include <alps/alea/util.hpp>

// TODO replace with <random>; kept for consistency through ALPS
#include <boost/random.hpp>

namespace alps { namespace alea { namespace util {
    template <typename T> class var1_model;
    template <typename T> class var1_run;
}}}

namespace alps { namespace alea { namespace util {

/**
 * Linear vector autoregressive model (VAR(1)).
 *
 * Representation of a simple discrete-time Markov process in which the `t`-th
 * step is given by the random vector: [1]
 *
 *                   X(t) = phi0 + phi1 * X(t-1) + eps(t)    (t > 0)
 *
 * where `phi0` is a shift vector, `phi` is the parameter matrix of the model,
 * and `eps(t)` is the shock term, which is Gaussian white noise with variances
 * `eps(t)[i] ~ N(0, var_eps[i])`.  (In the case of non-zero covariances, X
 * must be rotated into the eigenbasis of the covariance matrix.)
 *
 * The model will tend to a stationary solution with the following moments: [1]
 *
 *           E[X] = (1 - phi1)^{-1} phi0
 *     vec Cov[X] = (1 - phi1 (x) phi1)^{-1} vec Cov[eps]
 *         tau[X] = phi1 * (1 - phi1)^{-1}
 *
 * [1] https://www.kevinsheppard.com/images/5/56/Chapter5.pdf
 */
template <typename T>
class var1_model
{
public:
    typedef typename make_real<T>::type var_type;

public:
    var1_model() { }

    template <typename Der1, typename Der2, typename Der3>
    var1_model(const Eigen::MatrixBase<Der1> &phi0,
               const Eigen::MatrixBase<Der2> &phi1,
               const Eigen::MatrixBase<Der3> &var_eps)
        : phi0_(phi0)
        , phi1_(phi1)
        , var_eps_(var_eps)
        , stddev_eps_(var_eps.rows())
    {
        init();
    }

    /** Mean of the VAR(1) process */
    typename eigen<T>::col mean() const;

    /** Covariance of the process (not corrected for autocorrelation) */
    typename eigen<T>::matrix cov() const;

    /** Variance of the process (not corrected for autocorrelation) */
    typename eigen<var_type>::col var() const;

    /** Autocorrelation time matrix */
    typename eigen<T>::matrix ctau() const;

    /** Number of vector components */
    int size() const { return phi0_.rows(); }

    /** Start a new instance for this model (allows generating data) */
    var1_run<T> start() const { return var1_run<T>(*this); }

    /** Shift term of the model */
    const typename eigen<T>::col &phi0() const { return phi0_; }

    /** Linear lag term of the model */
    const typename eigen<T>::matrix &phi1() const { return phi1_; }

    /** Variance of the Gaussian noise that makes up the shock term. */
    const typename eigen<var_type>::col &var_eps() const { return var_eps_; }

    /** Standard dev of the Gaussian noise that makes up the shock term. */
    const typename eigen<var_type>::col &stddev_eps() const { return stddev_eps_; }

protected:
    void init();

private:
    typename eigen<T>::col phi0_;
    typename eigen<T>::matrix phi1_;
    typename eigen<var_type>::col var_eps_, stddev_eps_;
};

extern template class var1_model<double>;
extern template class var1_model<std::complex<double> >;

/**
 * One instance (time series) for linear vector autoregressive model (VAR(1)).
 *
 * @see var1_model<T>
 */
template <typename T>
class var1_run
{
public:
    typedef typename make_real<T>::type var_type;

public:
    var1_run();

    var1_run(const var1_model<T> &model);

    /** Reset the run to t=0 */
    void restart();

    /** Take the next step, given certain random input */
    template <typename RandomEngine>
    void step(RandomEngine &engine)
    {
        get_noise(engine);
        update();
    }

    /** Number of steps taken */
    size_t t() const { return t_; }

    /** Current position */
    const column<T> &xt() const { return xt_; }

    /** Reference to underlying model */
    const var1_model<T> &model() const { return *model_; }

protected:
    template <typename RandomEngine>
    void get_noise(RandomEngine &engine)
    {
        using boost::random::normal_distribution;
        for (size_t i = 0; i != epst_.size(); ++i) {
            normal_distribution<var_type> dist(0, model_->stddev_eps()[i]);
            epst_[i] = dist(engine);
        }
    }

    void update();

private:
    const var1_model<T> *model_;
    size_t t_;
    column<T> xt_;
    column<var_type> epst_;
};

extern template class var1_run<double>;
extern template class var1_run<std::complex<double> >;

}}}
