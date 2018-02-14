#include <alps/alea/util.hpp>
#include <alps/alea/internal/format.hpp>

namespace alps { namespace alea {

std::ostream &operator<<(std::ostream &stream, verbosity verb)
{
    internal::get_format(stream, PRINT_TERSE) = verb;
    return stream;
}

}}
