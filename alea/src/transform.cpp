#include <alps/alea/transform.hpp>

namespace alps { namespace alea {

template <typename T>
typename eigen<T>::matrix jacobian(const transform<T> &f,
                                   column<T> x, const column<T> &dx)
{
    size_t in_size = f.in_size();
    size_t out_size = f.out_size();

    typename eigen<T>::matrix result(out_size, out_size);
    for (size_t j = 0; j != in_size; ++j) {
        x(j) += dx(j);
        result.col(j) = f(x);
        x(j) -= dx(j);
    }
    result.colwise() -= f(x);
    result.array().colwise() /= dx.array();
    return result;
}

template eigen<double>::matrix jacobian(const transform<double> &,
                                        column<double>, const column<double> &);

}}
