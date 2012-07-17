/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2010 - 2012 by Lukas Gamper <gamperl@gmail.com>                   *
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

#include <boost/mpl/or.hpp>
#include <boost/integer.hpp>
#include <boost/functional/hash.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_pointer.hpp>
#include <boost/type_traits/is_integral.hpp>
#include <boost/type_traits/is_arithmetic.hpp>
#include <boost/type_traits/is_floating_point.hpp>

#include <functional>

namespace alps {

    template <typename T> inline typename boost::enable_if_c<
        boost::is_integral<T>::value, std::size_t
    >::type hash_value(T v) {
        // True random matrix from http://www.random.org/
        static boost::uint64_t const M[] = {
              0xc967ad408b3ff37dULL, 0xd0565464329edd93ULL, 0xfae44ec454bf6d2bULL, 0xfe026afc0848f820ULL
            , 0xf50747799b585df2ULL, 0x9a42a9fb2052e957ULL, 0x01ca087318209703ULL, 0x4ec19e57a48d71a6ULL
            , 0x689720ad7892bbc3ULL, 0x2b3a3664edb9fd9bULL, 0xa630d405bd054001ULL, 0x4e3c5211b69a6ce6ULL
            , 0x8390e4c71b4cabb2ULL, 0x543f41eae6c3ec02ULL, 0xb5700943efa1a562ULL, 0x96f303684e823465ULL
            , 0x0c2d87d6a1c1ed86ULL, 0xd1809c8131a3258bULL, 0xdd814b6d2f2a7c75ULL, 0x340478544a7b82a1ULL
            , 0x5ba62a6416224d81ULL, 0x0c907d73a0f0b59bULL, 0x43348db83e51a2efULL, 0xb139c834202c815fULL
            , 0xd30eebeefd252106ULL, 0x8cf2e815a28d30dfULL, 0xc4b5d412ec147890ULL, 0x05f2f015aa7ce0d7ULL
            , 0x207113672f494460ULL, 0x6c6d6cd444c52f6dULL, 0x9ddc8d38f03209cdULL, 0xc6950924be99e0b3ULL

            , 0xd67c9817a2a8dc0bULL, 0x09357cc73daa9d02ULL, 0x47c56953c156ebfdULL, 0xaca411e919aa9e6fULL
            , 0x05589249ac896aaaULL, 0x1c6c639d704ed162ULL, 0x7c6abd7e9a3ea933ULL, 0xb3ab8cca2a33ed58ULL
            , 0x361a9005d52c7a99ULL, 0x1a879aa8395205d4ULL, 0x514ca14039ebc32dULL, 0xb64a2b587c95c2baULL
            , 0x922a370511d8e089ULL, 0x60e3db4d9e53cdb6ULL, 0x360d4289212fcee3ULL, 0xf13d8ac513195e75ULL
            , 0x7bb153add1bc9450ULL, 0x919e2ed9e7756596ULL, 0x1ad949c54fba51dfULL, 0x2489554920a41e3fULL
            , 0xb733d8d447c4fc98ULL, 0xba046a71218924caULL, 0xe52ab1cfdb374fe7ULL, 0x4159be29f8d2ba15ULL
            , 0xa54605926d920368ULL, 0x259c2d36f57bfe14ULL, 0x8ee0255d05b96ceaULL, 0x88b135099e5360e4ULL
            , 0x4bae76467fdbe749ULL, 0x5ec80497cf5ed18aULL, 0xf8a5e0031171f368ULL, 0x985144d8f72f7cc7ULL
        };
        // TODO: consider using well or LFSR as hash funcitons, see http://www.lomont.org/Math/Papers/2008/Lomont_PRNG_2008.pdf

        boost::uint64_t h = 0, u = v;
        for (std::size_t i = 0; i < 64; ++i) {
            // get parity bits from http://graphics.stanford.edu/~seander/bithacks.html
            boost::uint64_t parity = u & M[i];
            parity ^= parity >> 1;
            parity ^= parity >> 2;
            parity = (parity & 0x1111111111111111ULL) * 0x1111111111111111ULL;
            h |= ((parity >> 60) & 1ULL) << i;
        }
        return h;
    }

    inline std::size_t hash_value(float v) {
        return hash_value(*reinterpret_cast<boost::uint_t<8 * sizeof(float)>::exact *>(&v));
    }

    inline std::size_t hash_value(double v) {
        return hash_value(*reinterpret_cast<boost::uint_t<8 * sizeof(double)>::exact *>(&v));
    }

    template <typename T> inline typename boost::enable_if_c<
        boost::is_pointer<T>::value, std::size_t
    >::type hash_value(T v) {
        return hash_value(reinterpret_cast<boost::uint_t<8 * sizeof(void *)>::exact>(v));
    }

    template <typename T> inline typename boost::disable_if<typename boost::mpl::or_<
          typename boost::is_arithmetic<T>::type
        , typename boost::is_pointer<T>::type
    >::type, T>::type hash_value(T v) {
        boost::hash<T> hasher;
        return hasher(v);
    }


    template <typename T> 
    struct hash : public std::unary_function<T, std::size_t> {
      /*!
Generates a hash. 
\return Returns an unsigned integer of type std::size_t.
       */
       std::size_t operator()(T v) const {
            return hash_value(v);
        }
    };

  /*!
Combines a hash with the hash of a value. 
   */
    template<typename T> void hash_combine(size_t & s, T const & v) {
        hash<T> hasher;
        s ^= hasher(v) + 0x6dbc79f65d57ddf9ULL + (s << 12) + (s >> 4);
    }

    template<typename I> inline void hash_range(std::size_t & s, I f, I l) {
        for(; f != l; ++f)
            hash_combine(s, *f);
    }

    template<typename I> inline std::size_t hash_range(I f, I l) {
        std::size_t s = 0;
        hash_range(s, f, l);
        return s;
    }
}
