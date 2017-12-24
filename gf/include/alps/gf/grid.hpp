/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPSCORE_GRID_HPP
#define ALPSCORE_GRID_HPP

#include <complex>
#include <cmath>
#include <vector>

namespace alps {
  namespace gf {
    /// Grids definitions for real frequency mesh
    namespace grid{

      /// Linear grid in real frequency
      class linear_real_frequency_grid {
      private:
        /// lowest frequency in real frequency space
        double emin_;
        /// highest frequency in real frequency space
        double emax_;
        /// number of frequency points
        int n_;
      public:
        linear_real_frequency_grid(double emin, double emax, int n) : emin_(emin), emax_(emax), n_(n){};
        void compute_points(std::vector<double> &points) const {
          points.resize(n_);
          double step = (emax_ - emin_)/double(n_-1);
          for(int i = 0; i<n_; i++) {
            points[i] = emin_ + step*i;
          }
        }
      };

      /// Logarithmic grid in real frequency
      class logarithmic_real_frequency_grid {
      private:
        /// first step value
        double t_min_;
        /// maximal positive shift from the central point of the grid
        double t_max_;
        /// central point of the grid
        double c_;
        /// number of frequency points
        int nfreq_;
      public:
        logarithmic_real_frequency_grid(double tmin, double tmax, double c, int n): t_min_(tmin), t_max_(tmax), c_(c), nfreq_(n) {
          if (tmin <= 0.0)
            throw std::invalid_argument("the parameter tmin must be greater than 0");
          if (tmax<tmin)
            throw std::invalid_argument("the parameter tmax must be greater than tmin");
        };
        logarithmic_real_frequency_grid(double tmin, double tmax, int n): t_min_(tmin), t_max_(tmax), c_(0), nfreq_(n) {
          if (tmin <= 0.0)
            throw std::invalid_argument("the parameter tmin must be greater than 0");
          if (tmax<tmin)
            throw std::invalid_argument("the parameter tmax must be greater than tmin");
        };
        void compute_points(std::vector<double> &points) const {
          points.resize(nfreq_);
          double scale = std::log(t_max_ / t_min_) / ((float) ((nfreq_ / 2 - 1)));
          points[nfreq_ / 2] = c_;
          for (int i = 0; i < nfreq_ / 2; ++i) {
            // check boundaries for an even # of frequencies
            if(i<nfreq_/2 - 1)
              points[nfreq_ / 2 + i + 1] = c_ + t_min_ * std::exp(((float) (i)) * scale);
            points[nfreq_ / 2 - i - 1] = c_ - t_min_ * std::exp(((float) (i)) * scale);
          }
          // if we have an odd # of frequencies, this catches the last element
          if (nfreq_ % 2 != 0)
            points[nfreq_ / 2 + nfreq_ / 2] = c_ + t_min_ * std::exp(((float) (nfreq_/2 - 1)) * scale);
        }
      };

      /// Quadratic grid in real frequency
      class quadratic_real_frequency_grid {
      private:
        /// number of frequency points
        int nfreq_;
        double spread_;
      public:
        quadratic_real_frequency_grid(double spread, int n): nfreq_(n) {
          if (spread < 1)
            throw std::invalid_argument("the parameter spread must be greater than 1");
          spread_ = spread;
        }
        void compute_points(std::vector<double> & points) const {
          points.resize(nfreq_);
          std::vector<double> temp(nfreq_);
          double t = 0;
          for (int i = 0; i < nfreq_; ++i) {
            double a = double(i) / (nfreq_ - 1);
            double factor = 4 * (spread_ - 1) * (a * a - a) + spread_;
            factor /= double(nfreq_ - 1) / (3. * (nfreq_ - 2))
                      * ((nfreq_ - 1) * (2 + spread_) - 4 + spread_);
            double delta_t = factor;
            t += delta_t;
            temp[i] = t;
          }
          points[nfreq_/2] = 0.;
          for (int i = 1; i <= nfreq_/2; ++i) {
            if(i<nfreq_/2)
              points[i + nfreq_ / 2] = temp[i - 1] / temp[nfreq_ / 2 - 1];
            points[nfreq_ / 2 - i] = -temp[i - 1] / temp[nfreq_ / 2 - 1];
          }
          //if we have an odd # of frequencies, this catches the last element
          if (nfreq_ % 2 != 0)
            points[nfreq_ / 2 + nfreq_ / 2] = temp[nfreq_/2 - 1] / temp[nfreq_ / 2 - 1];
        }
      };
    }
  }
}

#endif //ALPSCORE_GRID_HPP
