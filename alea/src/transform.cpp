#include <alps/alea/transform.hpp>

#include <iostream>

namespace alps { namespace alea {

template <typename T>
typename eigen<T>::matrix jacobian(const transform<T> &f, column<T> x, double dx)
{
    size_t in_size = f.in_size();
    size_t out_size = f.out_size();

    typename eigen<T>::matrix result(out_size, out_size);
    for (size_t j = 0; j != in_size; ++j) {
        x(j) += dx;
        result.col(j) = typename eigen<T>::col(f(x));   // FIXME
        x(j) -= dx;
    }
    result.colwise() -= f(x);
    result.array() /= dx;
    return result;
}

template eigen<double>::matrix jacobian(
            const transform<double> &, column<double>, double);
template eigen< std::complex<double> >::matrix jacobian(
            const transform<std::complex<double> > &, column<std::complex<double> >,
            double);

}}
