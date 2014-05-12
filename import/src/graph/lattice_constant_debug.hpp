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

#ifdef USE_LATTICE_CONSTANT_2D
#include <alps/graph/lattice_constant_2d.hpp>
#define ALPS_GRAPH_LATTICE_CONSTANT
#endif // USE_LATTICE_CONSTANT_2D

#ifndef ALPS_GRAPH_LATTICE_CONSTANT
#define ALPS_GRAPH_LATTICE_CONSTANT

#include <alps/ngs/stacktrace.hpp>

#include <alps/lattice/graph_helper.h>
#include <alps/lattice/graphproperties.h>
#include <alps/numeric/vector_functions.hpp>
#include <alps/graph/canonical_properties.hpp>

#include <boost/array.hpp>
#include <boost/unordered_set.hpp>
#include <boost/functional/hash.hpp>

#include <deque>
#include <vector>
#include <cstring>
#include <algorithm>
#include <stdexcept>

#if !defined(USE_COMPRESSED_EMBEDDING) && !defined(USE_COMPRESSED_EMBEDDING2) && !defined(USE_GENERIC_EMBEDDING)
    #define USE_GENERIC_EMBEDDING
#endif

namespace alps {
    namespace graph {
    
        namespace detail {
        
#ifdef USE_COMPRESSED_EMBEDDING2

            template<boost::uint64_t N = 2> class compressed_set {

                public:
                    compressed_set(boost::uint64_t pfx)
                        : prefix(pfx)
                        , count(0)
                        , mask((0x01ul << prefix) - 1)
                        , mem(new boost::uint64_t[N * (0x01ul << prefix)])
//, avgwalk(0)
                    {
                        std::memset(mem, 0x00, (N * (0x01ul << prefix)) << 3);
                    }

                    ~compressed_set() {
                        delete[] mem;
                    }

                    bool insert(boost::array<boost::uint64_t, N> const & data) {
                        boost::uint64_t * local_mem = mem + ((hash(&data[0], prefix) ^ (data[0] & mask)) << 0x01);
                        for (boost::uint64_t offset = 0; offset < mask; ++offset, local_mem += N) {
                            if (local_mem == mem + (N * (0x01ul << prefix)))
                                local_mem = mem;
                            if ((*local_mem & mask) == 0) {
                                *local_mem = (offset + 1) | (data[0] & ~mask);
                                *(local_mem + 1) = data[1];
//avgwalk += offset; 
                                if (3 * (0x01ul << (prefix - 2)) < ++count)
                                    grow();
                                return true;
                            } else if ((*local_mem & mask) == offset + 1)
                                if ((data[0] & ~mask) == (*local_mem & ~mask) and data[1] == *(local_mem + 1))
                                    return false;
                        }
                    }

                    std::size_t inline size() {
                        return count;
                    }
//boost::uint64_t avgwalk;

                private:

                    compressed_set(compressed_set const &) {}
                    
                    void grow() {
//double avgdist = avgwalk / double(count);
                        boost::uint64_t * old_mem = mem;
                        boost::uint64_t old_mask = mask;
                        boost::array<boost::uint64_t, N> data;

                        ++prefix;
                        count = 0;
//avgwalk = 0;
                        mask = (0x01ul << prefix) - 1;
                        mem = new boost::uint64_t[N * (0x01ul << prefix)];
                        std::memset(mem, 0x00, (N * (0x01ul << prefix)) << 3);
                        for (boost::uint64_t index = 0; index < N * (0x01ul << (prefix - 1)); index += N)
                            if ((old_mem[index] & old_mask) > 0) {
                                boost::uint64_t pfx = (old_mem[index] & old_mask) - 1 < (index >> 0x01)
                                    ? (index >> 0x01) - (old_mem[index] & old_mask) + 1
                                    : (0x01ul << (prefix + 1)) + (index >> 0x01) - (old_mem[index] & old_mask) + 1
                                ;
                                data[0] = ((hash(old_mem + index, prefix - 1) ^ pfx) & old_mask) | (old_mem[index] & ~old_mask);
                                for (std::size_t i = 1; i < N; ++i)
                                    data[i] = old_mem[index + i];
                                insert(data);
                            }
                        delete[] old_mem;
//std::cout << "resize to " << prefix << ", avg dist " << avgdist << ", avg dist in resize " << avgwalk / double(count) << std::endl;
                    }

                    inline boost::uint64_t hash(boost::uint64_t const * data, boost::uint64_t size) {

                        // True random matrix from http://www.random.org/
                        static boost::uint64_t const M[] = {
                        // 0
                              0xC97FA6018641C00Aul, 0x313B9FF32EE78E22ul, 0xB09D333C35ECF598ul, 0x70EB494C225C8501ul
                            , 0x871FA0AFF0E8D971ul, 0x2F3E31A6B63EDB46ul, 0xDE4879F9668B2B02ul, 0x851577EB5AF6E261ul
                            , 0xC4C48F00C9ACDB1Eul, 0xB29FAF2B70496833ul, 0x2F2B210A26EC6235ul, 0xBC54FBFEEA10870Ful
                            , 0x3DE4470FA76FADD4ul, 0xEA727243ADCE484Aul, 0x8FB0B0296FD1BF0Ful, 0x62AB26C2A8C93022ul
                            , 0xD7F79D51CBD45CECul, 0xF0760BE05C5FE2A1ul, 0x723A2E7F99318BF0ul, 0xC1AD3D13A0E6A7B1ul
                            , 0xDE310EAB0E3F85E5ul, 0x153B02872EC87536ul, 0xCA0E60E906D5BE7Dul, 0xD709B4E8E3B8AECEul
                            , 0x1766D61C3D5AC711ul, 0x5A20F4B6C54D2D35ul, 0x6C5C1C81C140AAB1ul, 0x7B33F8FF92A86F9Bul
                            , 0xE673969E440276B2ul, 0x48DB6F52CE258BFFul, 0xB53A220AA6B97075ul, 0x31D8487CBD2C4D95ul
                            
                            , 0x53AFD93A6CFF517Aul, 0xF768C3583AC89881ul, 0x0E06ADCF996D63C6ul, 0x4AA38C9E14F142BFul
                            , 0x8B37AC8A52A2852Dul, 0x1CBCE21EF0FBF549ul, 0xC03DE80357920E20ul, 0x88ED31F900280AD3ul
                            , 0x33E31D2CFDABE8FAul, 0x350DACF0DCA11C18ul, 0x5EF2557FDAAFA066ul, 0xAFF8BBED27044BCBul
                            , 0x8DE6F399B1C7913Eul, 0xD6DBFBB1DF54D945ul, 0x34E64693797A3195ul, 0x807198687AF192A3ul
                            , 0xADCF66AC6BAC0231ul, 0x55A36D8DBDFE7DC7ul, 0x851BD2A69A58BE9Bul, 0x4FC5CE549DD4F3C7ul
                            , 0x24EBD40AF222EB10ul, 0x6B2F358DD7B82313ul, 0x1076500155512D8Eul, 0x9CFF7B3F0908F321ul
                            , 0x49CAEBD21819749Ful, 0x8BABC20EE1EF956Bul, 0x0E6567044B08E730ul, 0xB8EA31CA02E2EBD8ul
                            , 0x24CCF9E87B0A3170ul, 0xDB7EC00E58BF0A93ul, 0xE1842DDAE5F4F60Bul, 0x51A09EB6CBE08C73ul
                        // 1
                            , 0x76A01C52BC3C0BE4ul, 0x1B5F0F2D057F71B9ul, 0xD18B1D9E533C0CB5ul, 0x56156E6AF8512435ul
                            , 0x0A5D4C192F22BE51ul, 0xEE7B77DECD7E5FBBul, 0xB81DA6C0ABDED564ul, 0x2FD95C25EF2E84F3ul
                            , 0xD20B63371060950Aul, 0x05BF8229D20EE4BEul, 0x9469363303E252E5ul, 0x47ABC49AF0944093ul
                            , 0x6EBCCBB211E1303Dul, 0xA8B25CC288AB30F7ul, 0xA6F20D074BE0E53Cul, 0xC572A8CFDB8D38C7ul
                            , 0x07A5D89F2DD67187ul, 0x3A179683DF534BFEul, 0xE020C675A7256B8Cul, 0xDE2F8C4E1E000057ul
                            , 0x5468254A15D428D1ul, 0x7528043E65F05D3Dul, 0x9C6942C5FF15C4CDul, 0x3BBD92D4A339DA3Eul
                            , 0x3E751F3B428982DAul, 0x77CFF15666BD2B83ul, 0x82ED8D3B79ACB366ul, 0x2E5E3D2DC197AEC0ul
                            , 0xFA3A06F649BD1F9Bul, 0x840193FF2CE86EB7ul, 0x8304E4C9C577ACC3ul, 0xB45AB7D1D2BCA76Cul
                            
                            , 0x96CED10A1CBF445Bul, 0xB66AE4450CFD7F80ul, 0xE868AD776D8E142Ful, 0x200F3C0C646ABCB7ul
                            , 0x2EFC98666892D9BBul, 0x7D062710E19A5226ul, 0x4926BC177175DD25ul, 0x8CAC36CD483E039Aul
                            , 0x619C6E66D1844598ul, 0x724EBA865E0CC26Ful, 0x5990F03174BD7357ul, 0x8924F932AD4953ADul
                            , 0xCAF667984EFF1420ul, 0x18B7D087A6F07005ul, 0x925639734B5BC90Aul, 0xB2B8305A0848C292ul
                            , 0x4305847902117C64ul, 0x6F9C20B04713FEE5ul, 0x027B23CE27C9E610ul, 0x94012D1E24BA78DCul
                            , 0xDB895749C286CE90ul, 0x9C212D17656508F7ul, 0x33B0DB9EE39781A8ul, 0x00A4DBBB8718A208ul
                            , 0xDEC29F502348E3F2ul, 0x83200373A93FFAE7ul, 0x7B2C64D05ECE1E83ul, 0xBDC05535507107F0ul
                            , 0x2FCA4423D764F55Eul, 0xE0E648ECB55EC69Eul, 0x539D44871ECB83CBul, 0xD0B9BE6A4C158487ul
                        // 2
                            , 0x48610FD11C1DB3E5ul, 0xC5A174889472FD93ul, 0x8F7EBF47D75DB381ul, 0x96A39A514899F47Dul
                            , 0x26088B90FA494924ul, 0x1B75FB27CF95097Dul, 0x36261AD1E8808208ul, 0x24B79DA120008A56ul
                            , 0x9FF8E37EE195A127ul, 0x40212D15E29EABA4ul, 0x1234EB18A7917586ul, 0xF3318C2F8A342B4Eul
                            , 0x68AB527DD3E333CFul, 0x3C28AE8C707C02E8ul, 0xA53DE1E639E4763Bul, 0x72721305AE6B5381ul
                            , 0x3C7ECDA5FDE10301ul, 0xC1C3FACE4175B9EFul, 0xEEA8F31E73629A02ul, 0x13F4E4C9C555F4DAul
                            , 0x49CF612BF14F1217ul, 0x90235C6473CF89EEul, 0xD6EBD44376CAE32Cul, 0xF2CD74AA58BFD4CBul
                            , 0x1BE1023C60E1B6A1ul, 0xC607D7634E04D3BFul, 0xDE8F2592459924FAul, 0xF259DE1A88A82BB1ul
                            , 0x00219ADB593B7C4Dul, 0x6E6656968DA254E5ul, 0xA656FF30917A0BF9ul, 0x088CC623512FF4A3ul

                            , 0x3C208D3767A208AEul, 0x0A437E8EE6890350ul, 0xE36C71F4DEFACA71ul, 0xB2F7AD92BD88B9BEul
                            , 0xC3E987ACDA1081C9ul, 0x6DAAA62CA600CBB8ul, 0xA333FD048ACD03F7ul, 0x1B966F77E3DC0E49ul
                            , 0xF22089C328BC681Ful, 0xEFD8FBA9021081B4ul, 0x4049216AE72FCA92ul, 0xC2075B46AFF343F7ul
                            , 0x2674CE72032F2F6Aul, 0x603CF0AA38419155ul, 0x5E5E05E48DD13963ul, 0xCE55082FA967FF08ul
                            , 0x85E8E7C2577F75DCul, 0xF73B2118A093B1B7ul, 0x9F0432F12E73AC46ul, 0xE387707F4F82EBEFul
                            , 0xA905976BC894BA1Aul, 0x04E546ABC88840BDul, 0xEB21423FB905AA25ul, 0xB0ABC82AED4BB1CAul
                            , 0xB91A4A0D7A249461ul, 0x56874CFEAB10C533ul, 0x403CD26A3A7CDAC3ul, 0x4F773EB21817DDBEul
                            , 0xBB20BD8E0B4949CAul, 0x41CAF02B93905DD2ul, 0x33C2280CF551FB7Bul, 0x793EFFEE831129EEul
                        // 3
                            , 0x330EA6666FE93B47ul, 0x269026B961682EEDul, 0xE0B18CCCF0575073ul, 0xF4A574727A9B33FDul
                            , 0xA42CDAEFF0E6E079ul, 0x9AD279C99475A395ul, 0x58D731293E5766D3ul, 0xA2EDA808B1A9970Ful
                            , 0x4DAC7A68770E0FE2ul, 0xB54485B4D031CE10ul, 0xA2B3FEFC7718EEF3ul, 0xB7EFA01DBCE08C6Ful
                            , 0x63DCE1AB2C189DE0ul, 0x2136870CFFFEBBFCul, 0xA6E91449297B627Eul, 0x0B433441BD5510A8ul
                            , 0xDF21422DC45A5E7Bul, 0xBD33180D7CE34B71ul, 0xF33D2CB37121B979ul, 0xFF450A483E532A4Bul
                            , 0xAA935913312FA568ul, 0x0E551629404AF3FEul, 0x8A3FC66C7EFEBDA9ul, 0x74560F7B3970ED3Cul
                            , 0xB6CC5808D7CCC335ul, 0xB51232F2628F7DB9ul, 0x589730D8AE17BBEDul, 0x832860166CB0925Eul
                            , 0x02C27229622FA772ul, 0xA6673CB443638FA2ul, 0x7F69085C0B0E4723ul, 0x3B855683F93CB4A1ul
                            
                            , 0xC8BC3FC70182D744ul, 0xCC1C002E347891B4ul, 0xC59072952162396Cul, 0x4F534B5B2E86F6FEul
                            , 0xF1412F84227BF4EEul, 0x736ADFE9315522ABul, 0x14E0611016CDCD23ul, 0x7D2777C26A777A4Eul
                            , 0x29FE75CD9D2D19C4ul, 0x73AD3E2DAAE16CA6ul, 0x58C9A4849BE2C20Ful, 0xDC3B1AA8A1A88D0Dul
                            , 0x6CC51EE38B95B718ul, 0x0B556C8A881A0B49ul, 0xF5197DFFFFB815C4ul, 0x7CE34F529DBDCB5Eul
                            , 0xB021AFEDB95E87F3ul, 0x156E08DF64CAC366ul, 0xE10593835674623Eul, 0xB766AC8416A733EAul
                            , 0x32C6DB06EC577C82ul, 0x321B146863E55773ul, 0x12DC023031BE3EB5ul, 0x0EA6CC2DA419BA75ul
                            , 0x8A3CD79A072C00BEul, 0x19984AC7BE3A21C9ul, 0x17D6FE0E387FE617ul, 0x8DB42792C013F6C3ul
                            , 0xE0E2BA21721A012Aul, 0x78046BBE3904B1B0ul, 0xCA2B5C0B6B955D42ul, 0xDF02C85C1DC88E05ul
                        };

                        // TODO: check other parity counts, maybe it can be done faster ...
                        // get parity bits from http://graphics.stanford.edu/~seander/bithacks.html#ParityNaive
                        boost::uint64_t hash = 0;
                        boost::uint64_t data_0 = data[0] & ~((0x01ul << size) - 1);
                        for (boost::uint64_t i = 0; i < size; ++i) {
                            boost::uint64_t parity = (data_0 & M[i]) ^ (data[1] & M[64 + i]);
                            parity ^= parity >> 1;
                            parity ^= parity >> 2;
                            parity = (parity & 0x1111111111111111ul) * 0x1111111111111111ul;
                            hash |= ((parity >> 60) & 0x01ul) << i;
                        }
                        return hash;
                    }

                    boost::uint64_t prefix;
                    boost::uint64_t count;
                    boost::uint64_t mask;
                    boost::uint64_t * mem;
            };
#endif

#if defined(USE_GENERIC_EMBEDDING) || defined(CHECK_COMPRESSED_EMBEDDING)
            struct embedding_generic_type {

                embedding_generic_type(std::size_t vertices_size, std::size_t edges_size)
                    : hash(0)
                    , counter(new boost::uint8_t(1))
                    , vertices(new std::vector<std::vector<boost::uint16_t> >(vertices_size))
                    , edges(new std::vector<boost::uint64_t>((edges_size >> 6) + ((edges_size & 0x3F) == 0 ? 0 : 1)))
                {}
                
                embedding_generic_type(embedding_generic_type const & rhs)
                    : hash(rhs.hash)
                    , counter(rhs.counter)
                    , vertices(rhs.vertices)
                    , edges(rhs.edges)
#ifdef CHECK_COMPRESSED_EMBEDDING
                    , pinning(rhs.pinning)
                    , occCnt(rhs.occCnt)
#endif
                {
                    assert(*counter < boost::integer_traits<boost::uint8_t>::const_max - 1);
                    ++*counter;
                }

                ~embedding_generic_type() {
                    if (!--*counter) {
                        delete counter;
                        delete vertices;
                        delete edges;
                    }
                }

                bool operator == (embedding_generic_type const & rhs) const {
                    return hash == rhs.hash
                        && *edges == *rhs.edges
                        && *vertices == *rhs.vertices
                    ;
                }

                std::size_t hash;
                boost::uint8_t * counter;
                std::vector<std::vector<boost::uint16_t> > * vertices;
                std::vector<boost::uint64_t> * edges;                
                
#ifdef CHECK_COMPRESSED_EMBEDDING
                std::vector<unsigned> pinning;
                int occCnt;
#endif
                private:
                    embedding_generic_type() {}
            };

            std::size_t hash_value(embedding_generic_type const & value) {
                return value.hash;
            }
#endif
        
#if defined(USE_COMPRESSED_EMBEDDING) || defined(CHECK_COMPRESSED_EMBEDDING)
            struct embedding_2d_type {
                embedding_2d_type() {
                    std::memset(data.c_array(), 0, data.size());
                }

                embedding_2d_type(embedding_2d_type const & rhs) 
#ifdef CHECK_COMPRESSED_EMBEDDING
                    : pinning(rhs.pinning)
                    , occCnt(rhs.occCnt)
#endif
                {
                    std::memcpy(data.c_array(), &(rhs.data[0]), data.size());
                }
                
                bool operator == (embedding_2d_type const & rhs) const {
                    return !memcmp(&(data[0]), &(rhs.data[0]), data.size());
                }
                boost::array<boost::uint8_t, 10> data;

#ifdef CHECK_COMPRESSED_EMBEDDING
                std::vector<unsigned> pinning;
                int occCnt;
#endif

            };

            std::size_t hash_value(embedding_2d_type const & value) {
                using boost::hash_range;
                return hash_range(value.data.begin(), value.data.end());
            }
#endif

            template <typename Graph, typename Lattice> void build_translation_table(
                  Graph const & graph
                , Lattice const & lattice
                , std::vector<std::vector<boost::uint_t<8>::fast> > & distance_to_boarder
            ) {
                typedef typename alps::lattice_traits<Lattice>::cell_iterator cell_iterator;
                typedef typename alps::lattice_traits<Lattice>::offset_type offset_type;
                typedef typename alps::lattice_traits<Lattice>::size_type cell_index_type;

                std::vector<std::vector<unsigned> > translations(dimension(lattice), std::vector<unsigned>(num_vertices(graph), num_vertices(graph)));
                unsigned vtcs_per_ucell = num_vertices(alps::graph::graph(unit_cell(lattice)));
                for(std::size_t d = 0; d < dimension(lattice); ++d) {
                    for(std::pair<cell_iterator,cell_iterator> c = cells(lattice); c.first != c.second; ++c.first) {
                        offset_type ofst = offset(*c.first,lattice);
                        offset_type move(dimension(lattice));
                        move[d] = -1;
                        std::pair<bool,bool> on_lattice_pbc_crossing = shift(ofst,move,lattice);
                        if(on_lattice_pbc_crossing.first && !on_lattice_pbc_crossing.second) {
                            const cell_index_type cellidx = index(*c.first,lattice);
                            const cell_index_type neighboridx = index(cell(ofst, lattice), lattice);
                            for(unsigned v = 0; v < vtcs_per_ucell; ++v)
                                translations[d][cellidx * vtcs_per_ucell + v] = neighboridx * vtcs_per_ucell + v;
                        }
                    }
                    unsigned v;
                    for (std::vector<unsigned>::const_iterator it = translations[d].begin(); it != translations[d].end(); ++it) {
                        if (*it != num_vertices(graph))
                        {
                            distance_to_boarder[d][v = *it] = 0;
                            while ((v = translations[d][v]) != num_vertices(graph))
                                ++distance_to_boarder[d][*it];
                        }
                    }
                }
            }
            
            struct embedding_found {};

#ifdef USE_COMPRESSED_EMBEDDING2
            template<typename Subgraph, typename Graph, unsigned SubVertexNum, unsigned CoordNum> void lattice_constant_embedding(
                  std::size_t s
                , boost::uint64_t & visited
                , std::size_t & index
                , boost::array<boost::uint64_t, 2> & embedding
                , Subgraph const & S
                , std::vector<std::vector<boost::uint_t<8>::fast> > const & distance_to_boarder
                , std::vector<typename boost::graph_traits<Graph>::vertex_descriptor> const & pinning
            ) {
                boost::array<boost::uint8_t, 2 * CoordNum> stack;
                std::memset(stack.c_array(), SubVertexNum, 2 * CoordNum);
                typename boost::graph_traits<Subgraph>::adjacency_iterator s_ai, s_ae;
                for (boost::tie(s_ai, s_ae) = adjacent_vertices(s, S); s_ai != s_ae; ++s_ai)
                    // TODO: make expression tempalte ...
                    for (boost::uint_t<8>::fast j = 0; j < CoordNum; ++j)
                        if (int(distance_to_boarder[j][pinning[*s_ai]]) - int(distance_to_boarder[j][pinning[s]]) == -1) {
                            embedding[(2 * index * CoordNum + j) >> 6] |= 0x01ul << ((2 * index * CoordNum + j) & 0x3F);
                            stack[2 * j] = *s_ai;
                        } else if (int(distance_to_boarder[j][pinning[*s_ai]]) - int(distance_to_boarder[j][pinning[s]]) == 1) {
                            embedding[(2 * index * CoordNum + CoordNum + j) >> 6] |= 0x01ul << ((2 * index * CoordNum + CoordNum + j) & 0x3F);
                            stack[2 * j + 1] = *s_ai;
                        }
                visited |= 0x01ul << s;
                ++index;
                // TODO: make expression template
                for (boost::uint8_t * it = stack.c_array(); it != stack.c_array() + 2 * CoordNum; ++it)
                    if (*it < SubVertexNum and !(visited & (0x01ul << *it)))
                        lattice_constant_embedding<Subgraph, Graph, SubVertexNum, CoordNum>(*it, visited, index, embedding, S, distance_to_boarder, pinning);
            }
#endif
#if defined(USE_COMPRESSED_EMBEDDING) || defined(CHECK_COMPRESSED_EMBEDDING)
            template<typename Subgraph, typename Graph, unsigned SubVertexNum, unsigned CoordNum> void lattice_constant_embedding(
                  std::size_t s
                , boost::dynamic_bitset<> & visited
                , std::size_t & index
                , boost::array<boost::uint8_t, 10> & data
                , Subgraph const & S
                , std::vector<std::vector<boost::uint_t<8>::fast> > const & distance_to_boarder
                , std::vector<typename boost::graph_traits<Graph>::vertex_descriptor> const & pinning
            ) {
                boost::array<boost::uint8_t, 2 * CoordNum> stack;
                std::memset(stack.c_array(), SubVertexNum, 2 * CoordNum);
                typename boost::graph_traits<Subgraph>::adjacency_iterator s_ai, s_ae;
                for (boost::tie(s_ai, s_ae) = adjacent_vertices(s, S); s_ai != s_ae; ++s_ai)
                    for (boost::uint_t<8>::fast j = 0; j < CoordNum; ++j)
                        if (int(distance_to_boarder[j][pinning[*s_ai]]) - int(distance_to_boarder[j][pinning[s]]) == -1) {
                            data[(2 * index * CoordNum + j) >> 3] |= 0x01 << ((2 * index * CoordNum + j) & 0x07);
                            stack[2 * j] = *s_ai;
                        } else if (int(distance_to_boarder[j][pinning[*s_ai]]) - int(distance_to_boarder[j][pinning[s]]) == 1) {
                            data[(2 * index * CoordNum + CoordNum + j) >> 3] |= 0x01 << ((2 * index * CoordNum + CoordNum + j) & 0x07);
                            stack[2 * j + 1] = *s_ai;
                        }
                visited[s] = true;
                ++index;
                // TODO: make expression template
                for (typename boost::array<boost::uint8_t, 2 * CoordNum>::const_iterator it = stack.begin(); it != stack.end(); ++it)
                    if (*it < SubVertexNum and !visited[*it])
                        lattice_constant_embedding<Subgraph, Graph, SubVertexNum, CoordNum>(*it, visited, index, data, S, distance_to_boarder, pinning);
            }
#endif

            // TODO: move back into main function after optimizing
            template<typename Subgraph, typename Graph, unsigned SubVertexNum, unsigned CoordNum> void lattice_constant_insert(
                  Subgraph const & S
                , Graph const & G
                , std::vector<std::size_t> const & I
#ifdef USE_COMPRESSED_EMBEDDING2
                , compressed_set<> & matches
#endif
#if defined(USE_COMPRESSED_EMBEDDING) || defined(CHECK_COMPRESSED_EMBEDDING)
                , boost::unordered_set<embedding_2d_type> & matches_2d
#endif
#if defined(USE_GENERIC_EMBEDDING) || defined(CHECK_COMPRESSED_EMBEDDING)
                , boost::unordered_set<embedding_generic_type> & matches_generic
#endif
                , std::vector<std::vector<boost::uint_t<8>::fast> > const & distance_to_boarder
                , std::vector<typename boost::graph_traits<Graph>::vertex_descriptor> const & pinning
                , typename partition_type<Subgraph>::type const & subgraph_orbit
                , std::size_t unit_cell_size
                , boost::mpl::true_
            ) {
                throw embedding_found();
            }

            // TODO: move back into main function after optimizing
            template<typename Subgraph, typename Graph, unsigned SubVertexNum, unsigned CoordNum> void lattice_constant_insert(
                  Subgraph const & S
                , Graph const & G
                , std::vector<std::size_t> const & I
#ifdef USE_COMPRESSED_EMBEDDING2
                , compressed_set<> & matches
#endif
#if defined(USE_COMPRESSED_EMBEDDING) || defined(CHECK_COMPRESSED_EMBEDDING)
                , boost::unordered_set<embedding_2d_type> & matches_2d
#endif
#if defined(USE_GENERIC_EMBEDDING) || defined(CHECK_COMPRESSED_EMBEDDING)
                , boost::unordered_set<embedding_generic_type> & matches_generic
#endif
                , std::vector<std::vector<boost::uint_t<8>::fast> > const & distance_to_boarder
                , std::vector<typename boost::graph_traits<Graph>::vertex_descriptor> const & pinning
                , typename partition_type<Subgraph>::type const & subgraph_orbit
                , std::size_t unit_cell_size
                // TODO: make argument, to pass SubVertexNum and CoordNum, so no explicit call is needed ...
                , boost::mpl::false_
            ) {

#ifdef USE_COMPRESSED_EMBEDDING2

                // TODO: figure out this number, only 40 bits are used per graph, only use one uint64_t!
                boost::array<boost::uint64_t, 2> embedding;
                std::memset(embedding.c_array(), 0x00, embedding.size() << 3);

                std::size_t start = 0;
                for (std::size_t i = 1; i < num_vertices(S); ++i) {
                    std::size_t dist_start = 0, dist_i = 0;
                    for (std::size_t j = 0; j < CoordNum; ++j) {
                        dist_start += distance_to_boarder[j][pinning[start]];
                        dist_i += distance_to_boarder[j][pinning[i]];
                    }
                    if (dist_i < dist_start)
                        start = i;
                    else if (dist_i == dist_start)
                        for (std::size_t j = 0; j < CoordNum; ++j) {
                            if (distance_to_boarder[j][pinning[i]] == distance_to_boarder[j][pinning[start]])
                                continue;
                            else if (distance_to_boarder[j][pinning[i]] < distance_to_boarder[j][pinning[start]])
                                start = i;
                            break;
                        }
                }
                // TODO: use uint64_t instead of bitset
                boost::uint64_t visited = 0x00;
                std::size_t index = 0;
                lattice_constant_embedding<Subgraph, Graph, SubVertexNum, CoordNum>(start, visited, index, embedding, S, distance_to_boarder, pinning);
                matches.insert(embedding);
#endif
            
#if defined(USE_COMPRESSED_EMBEDDING) || defined(CHECK_COMPRESSED_EMBEDDING)

#ifdef CHECK_COMPRESSED_EMBEDDING
                // if S has more than 20 vertices, chane this
                boost::array<boost::uint8_t, 20> ordred_vertices_chk;
                std::memset(ordred_vertices_chk.c_array(), boost::uint8_t(num_vertices(S)), ordred_vertices_chk.size());
                boost::array<boost::uint8_t, 20> vertices_order;
                boost::array<boost::uint_t<32>::fast, 20> ordred_edges;
                std::memset(ordred_edges.c_array(), 0, num_edges(S));

                // If the lattice has more than two dimensions, change that
                for (boost::uint_t<8>::fast i = 0; i < pinning.size(); ++i) {
                    // TODO: this can be done faster ...
                    boost::uint_t<8>::fast index = 0, vertex = i;
                    // TODO: make a lookup table for that and pass the reference at each call ...
                    for (std::size_t i = 0; i < I[vertex]; ++i)
                        index += subgraph_orbit[i].size();
                    for (; ordred_vertices_chk[index] != boost::uint8_t(num_vertices(S)); ++index)
                        if (
                               distance_to_boarder[0][pinning[vertex]] < distance_to_boarder[0][pinning[ordred_vertices_chk[index]]]
                            or (
                                    distance_to_boarder[0][pinning[vertex]] == distance_to_boarder[0][pinning[ordred_vertices_chk[index]]]
                                and distance_to_boarder[1][pinning[vertex]] < distance_to_boarder[1][pinning[ordred_vertices_chk[index]]]
                            )
                        )
                            std::swap(vertex, ordred_vertices_chk[index]);
                    ordred_vertices_chk[index] = vertex;
                }
                
                for (boost::uint_t<8>::fast i = 0; i < ordred_vertices_chk.size(); ++i)
                    vertices_order[ordred_vertices_chk[i]] = i;

                std::size_t pos = 0;
                typename boost::graph_traits<Subgraph>::edge_iterator ei, ee;
                for (boost::tie(ei, ee) = edges(S); ei != ee; ++ei, ++pos) {
                    boost::uint_t<8>::fast vs = source(*ei, S), vt = target(*ei, S);
                    if (vertices_order[vs] < vertices_order[vt])
                        std::swap(vs, vt);
                    ordred_edges[pos] = (vertices_order[vs] << 10) + (vertices_order[vt] << 2) + (distance_to_boarder[0][pinning[vs]] != distance_to_boarder[0][pinning[vt]]
                        ? (distance_to_boarder[0][pinning[vs]] < distance_to_boarder[0][pinning[vt]] ? 0x00 : 0x01)
                        : (distance_to_boarder[1][pinning[vs]] < distance_to_boarder[1][pinning[vt]] ? 0x02 : 0x03)
                    );
                }
                std::sort(ordred_edges.begin(), ordred_edges.begin() + num_edges(S));
#endif
                embedding_2d_type embedding_2d;
                // if S has more than 20 vertices, chane this
                boost::array<std::size_t, SubVertexNum> ordred_vertices;
                // TODO: this can be done much faster ...
                
#ifdef CHECK_COMPRESSED_EMBEDDING
                static int occCnt = 0;
                ++occCnt;
#endif
                
                std::size_t start = 0;
                for (std::size_t i = 1; i < num_vertices(S); ++i) {
                    std::size_t dist_start = 0, dist_i = 0;
                    for (std::size_t j = 0; j < CoordNum; ++j) {
                        dist_start += distance_to_boarder[j][pinning[start]];
                        dist_i += distance_to_boarder[j][pinning[i]];
                    }
                    if (dist_i < dist_start)
                        start = i;
                    else if (dist_i == dist_start)
                        for (std::size_t j = 0; j < CoordNum; ++j) {
                            if (distance_to_boarder[j][pinning[i]] == distance_to_boarder[j][pinning[start]])
                                continue;
                            else if (distance_to_boarder[j][pinning[i]] < distance_to_boarder[j][pinning[start]])
                                start = i;
                            break;
                        }
                }
                boost::dynamic_bitset<> visited(num_vertices(S));
                std::size_t index = 0;
                lattice_constant_embedding<Subgraph, Graph, SubVertexNum, CoordNum>(start, visited, index, embedding_2d.data, S, distance_to_boarder, pinning);
                            
#ifdef CHECK_COMPRESSED_EMBEDDING
                embedding_2d.pinning.resize(pinning.size());
                std::copy(pinning.begin(), pinning.end(), embedding_2d.pinning.begin());
                embedding_2d.occCnt = occCnt;
#else
                matches_2d.insert(embedding_2d);
#endif

#endif
#if defined(USE_GENERIC_EMBEDDING) || defined(CHECK_COMPRESSED_EMBEDDING)

                embedding_generic_type embedding_generic(subgraph_orbit.size(), num_vertices(S) * (num_vertices(S) + 1) / 2);
            
                for (std::vector<std::vector<boost::uint16_t> >::iterator it = embedding_generic.vertices->begin(); it != embedding_generic.vertices->end(); ++it)
                    it->reserve(subgraph_orbit[it - embedding_generic.vertices->begin()].size());

                std::size_t bits_per_dim = 0;
                while ((0x01 << ++bits_per_dim) < num_vertices(S));
                assert((0x01 << (distance_to_boarder.size() * bits_per_dim)) < boost::integer_traits<boost::uint16_t>::const_max);

                std::vector<boost::uint_t<8>::fast> distances(distance_to_boarder.size(), num_vertices(G));
                for (typename std::vector<typename boost::graph_traits<Graph>::vertex_descriptor>::const_iterator it = pinning.begin(); it != pinning.end(); ++it)
                    for(std::size_t d = 0; d < distance_to_boarder.size(); ++d)
                        distances[d] = std::min(distances[d], distance_to_boarder[d][*it]);
                std::vector<boost::uint16_t> lattice_pinning(pinning.size());
                for (typename std::vector<typename boost::graph_traits<Graph>::vertex_descriptor>::const_iterator it = pinning.begin(); it != pinning.end(); ++it) {
                    lattice_pinning[it - pinning.begin()] = *it % unit_cell_size;
                    for(std::size_t d = 0; d < distance_to_boarder.size(); ++d) {
                        lattice_pinning[it - pinning.begin()] <<= bits_per_dim;
                        lattice_pinning[it - pinning.begin()] += distance_to_boarder[d][*it] - distances[d];
                    }
                    (*embedding_generic.vertices)[I[it - pinning.begin()]].push_back(lattice_pinning[it - pinning.begin()]);
                }
                for (std::vector<std::vector<boost::uint16_t> >::iterator it = embedding_generic.vertices->begin(); it != embedding_generic.vertices->end(); ++it) {
                    using boost::hash_combine;
                    std::sort(it->begin(), it->end());
                    for (std::vector<boost::uint16_t>::const_iterator jt = it->begin(); jt != it->end(); ++jt)
                        hash_combine(embedding_generic.hash, *jt);
                }

                for (std::vector<boost::uint16_t>::iterator it = lattice_pinning.begin(); it != lattice_pinning.end(); ++it) {
                    std::vector<boost::uint16_t>::iterator jt = (*embedding_generic.vertices)[I[it - lattice_pinning.begin()]].begin();
                    for (; *jt != *it; ++jt);
                    *it = jt - (*embedding_generic.vertices)[I[it - lattice_pinning.begin()]].begin();
                    for (std::size_t i = 0; i < I[it - lattice_pinning.begin()]; ++i)
                        *it += (*embedding_generic.vertices)[i].size();
                }

                typename boost::graph_traits<Subgraph>::edge_iterator s_ei, s_ee;
                for (boost::tie(s_ei, s_ee) = edges(S); s_ei != s_ee; ++s_ei) {
                    std::size_t v1 = std::min(lattice_pinning[source(*s_ei, S)], lattice_pinning[target(*s_ei, S)]);
                    std::size_t v2 = std::max(lattice_pinning[source(*s_ei, S)], lattice_pinning[target(*s_ei, S)]);
                    std::size_t index = v1 * num_vertices(S) - (v1 - 1) * v1 / 2 + v2 - v1;
                    (*embedding_generic.edges)[index >> 6] |= 0x01 << (index & 0x3F);
                }
                for (std::vector<boost::uint64_t>::const_iterator it = embedding_generic.edges->begin(); it != embedding_generic.edges->end(); ++it) {
                    using boost::hash_combine;
                    hash_combine(embedding_generic.hash, *it);
                }
                
#ifdef CHECK_COMPRESSED_EMBEDDING
                embedding_generic.pinning.resize(pinning.size());
                std::copy(pinning.begin(), pinning.end(), embedding_generic.pinning.begin());
                embedding_generic.occCnt = occCnt;
#else
                matches_generic.insert(embedding_generic);
#endif

#endif

#ifdef CHECK_COMPRESSED_EMBEDDING
                {
                    using boost::hash_range;

                    boost::unordered_set<embedding_2d_type>::const_iterator it_2d;
                    boost::unordered_set<embedding_generic_type>::const_iterator it_generic;
                    bool b_2d, b_generic;
                    boost::tie(it_2d, b_2d) = matches_2d.insert(embedding_2d);
                    boost::tie(it_generic, b_generic) = matches_generic.insert(embedding_generic);
                    
                    if (b_2d and !b_generic) {
                        std::cout << "inserted in compressed " << occCnt << ", not in generic " << it_generic->occCnt << std::endl;

                        {
                            boost::array<boost::uint_t<8>::fast, 2> min_dist_to_boarder = { {
                                  boost::integer_traits<boost::uint8_t>::const_max - 1
                                , boost::integer_traits<boost::uint8_t>::const_max - 1
                            } };
                            boost::array<boost::uint_t<8>::fast, 2> max_dist_to_boarder = { { 0, 0} };
                            for (boost::uint_t<8>::fast i = 0; i < it_generic->pinning.size(); ++i) {
                                min_dist_to_boarder[0] = std::min(min_dist_to_boarder[0], distance_to_boarder[0][it_generic->pinning[i]]);
                                min_dist_to_boarder[1] = std::min(min_dist_to_boarder[1], distance_to_boarder[1][it_generic->pinning[i]]);
                                max_dist_to_boarder[0] = std::max(max_dist_to_boarder[0], distance_to_boarder[0][it_generic->pinning[i]]);
                                max_dist_to_boarder[1] = std::max(max_dist_to_boarder[1], distance_to_boarder[1][it_generic->pinning[i]]);
                            }
                            
                            for (std::size_t i = min_dist_to_boarder[0]; i <= max_dist_to_boarder[0]; ++i) {
                                for (std::size_t j = min_dist_to_boarder[1]; j <= max_dist_to_boarder[1] + num_vertices(S); ++j) {
                                    bool match = false;
                                    for (typename std::vector<unsigned>::const_iterator it = it_generic->pinning.begin(); it != it_generic->pinning.end(); ++it)
                                        if (distance_to_boarder[0][*it] == i && distance_to_boarder[1][*it] == j) {
                                            std::cout << (it - it_generic->pinning.begin()) << " ";
                                            match = true;
                                        }
                                    if (!match)
                                        std::cout << "  ";
                                }
                                std::cout << std::endl;
                            }                            
                        }
                        std::cout << std::endl;

                        {
                            boost::array<boost::uint_t<8>::fast, 2> min_dist_to_boarder = { {
                                  boost::integer_traits<boost::uint8_t>::const_max - 1
                                , boost::integer_traits<boost::uint8_t>::const_max - 1
                            } };
                            boost::array<boost::uint_t<8>::fast, 2> max_dist_to_boarder = { { 0, 0} };
                            for (boost::uint_t<8>::fast i = 0; i < pinning.size(); ++i) {
                                min_dist_to_boarder[0] = std::min(min_dist_to_boarder[0], distance_to_boarder[0][pinning[i]]);
                                min_dist_to_boarder[1] = std::min(min_dist_to_boarder[1], distance_to_boarder[1][pinning[i]]);
                                max_dist_to_boarder[0] = std::max(max_dist_to_boarder[0], distance_to_boarder[0][pinning[i]]);
                                max_dist_to_boarder[1] = std::max(max_dist_to_boarder[1], distance_to_boarder[1][pinning[i]]);
                            }
                            
                            for (std::size_t i = min_dist_to_boarder[0]; i <= max_dist_to_boarder[0]; ++i) {
                                for (std::size_t j = min_dist_to_boarder[1]; j <= max_dist_to_boarder[1] + num_vertices(S); ++j) {
                                    bool match = false;
                                    for (typename std::vector<typename boost::graph_traits<Graph>::vertex_descriptor>::const_iterator it = pinning.begin(); it != pinning.end(); ++it)
                                        if (distance_to_boarder[0][*it] == i && distance_to_boarder[1][*it] == j) {
                                            std::cout << (it - pinning.begin()) << " ";
                                            match = true;
                                        }
                                    if (!match)
                                        std::cout << "  ";
                                }
                                std::cout << std::endl;
                            }
                        }
                        
                    }
                        
                    
                    if (!b_2d and b_generic) {
                        std::cout << "inserted in generic " << occCnt << ", not in compressed " << it_2d->occCnt << std::endl;

                        for (std::size_t i = 0; i < subgraph_orbit.size(); ++i) {
                            for (std::size_t j = 0; j < subgraph_orbit[i].size(); ++j)
                                std::cout << subgraph_orbit[i][j] << " ";
                            std::cout << " | ";
                        }
                        std::cout << std::endl;
                        {
                            boost::array<boost::uint_t<8>::fast, 2> min_dist_to_boarder = { {
                                  boost::integer_traits<boost::uint8_t>::const_max - 1
                                , boost::integer_traits<boost::uint8_t>::const_max - 1
                            } };
                            boost::array<boost::uint_t<8>::fast, 2> max_dist_to_boarder = { { 0, 0} };
                            for (boost::uint_t<8>::fast i = 0; i < it_2d->pinning.size(); ++i) {
                                min_dist_to_boarder[0] = std::min(min_dist_to_boarder[0], distance_to_boarder[0][it_2d->pinning[i]]);
                                min_dist_to_boarder[1] = std::min(min_dist_to_boarder[1], distance_to_boarder[1][it_2d->pinning[i]]);
                                max_dist_to_boarder[0] = std::max(max_dist_to_boarder[0], distance_to_boarder[0][it_2d->pinning[i]]);
                                max_dist_to_boarder[1] = std::max(max_dist_to_boarder[1], distance_to_boarder[1][it_2d->pinning[i]]);
                            }
                            
                            for (std::size_t i = min_dist_to_boarder[0]; i <= max_dist_to_boarder[0]; ++i) {
                                for (std::size_t j = min_dist_to_boarder[1]; j <= max_dist_to_boarder[1] + num_vertices(S); ++j) {
                                    bool match = false;
                                    for (typename std::vector<unsigned>::const_iterator it = it_2d->pinning.begin(); it != it_2d->pinning.end(); ++it)
                                        if (distance_to_boarder[0][*it] == i && distance_to_boarder[1][*it] == j) {
                                            std::cout << (it - it_2d->pinning.begin()) << " ";
                                            match = true;
                                        }
                                    if (!match)
                                        std::cout << "  ";
                                }
                                std::cout << std::endl;
                            }                            
                        }
                        std::cout << std::endl;
                        for (std::size_t i = 0; i < num_vertices(S); ++i)
                            std::cout << unsigned(ordred_vertices[i]) << " ";
                        std::cout << std::endl;                        
                        {
                            boost::array<boost::uint_t<8>::fast, 2> min_dist_to_boarder = { {
                                  boost::integer_traits<boost::uint8_t>::const_max - 1
                                , boost::integer_traits<boost::uint8_t>::const_max - 1
                            } };
                            boost::array<boost::uint_t<8>::fast, 2> max_dist_to_boarder = { { 0, 0} };
                            for (boost::uint_t<8>::fast i = 0; i < pinning.size(); ++i) {
                                min_dist_to_boarder[0] = std::min(min_dist_to_boarder[0], distance_to_boarder[0][pinning[i]]);
                                min_dist_to_boarder[1] = std::min(min_dist_to_boarder[1], distance_to_boarder[1][pinning[i]]);
                                max_dist_to_boarder[0] = std::max(max_dist_to_boarder[0], distance_to_boarder[0][pinning[i]]);
                                max_dist_to_boarder[1] = std::max(max_dist_to_boarder[1], distance_to_boarder[1][pinning[i]]);
                            }
                            
                            for (std::size_t i = min_dist_to_boarder[0]; i <= max_dist_to_boarder[0]; ++i) {
                                for (std::size_t j = min_dist_to_boarder[1]; j <= max_dist_to_boarder[1] + num_vertices(S); ++j) {
                                    bool match = false;
                                    for (typename std::vector<typename boost::graph_traits<Graph>::vertex_descriptor>::const_iterator it = pinning.begin(); it != pinning.end(); ++it)
                                        if (distance_to_boarder[0][*it] == i && distance_to_boarder[1][*it] == j) {
                                            std::cout << (it - pinning.begin()) << " ";
                                            match = true;
                                        }
                                    if (!match)
                                        std::cout << "  ";
                                }
                                std::cout << std::endl;
                            }
                        }
                    }
                }
#endif
            }

            template<typename Subgraph, typename Graph> bool lattice_constant_vertex_equal(
                  typename boost::graph_traits<Subgraph>::vertex_descriptor const & s
                , typename boost::graph_traits<Graph>::vertex_descriptor const & g
                , Subgraph const & S
                , Graph const & G
                , boost::mpl::true_
            ) {
                return get(alps::vertex_type_t(), S)[s] == get(alps::vertex_type_t(), G)[g];
            } 

            template<typename Subgraph, typename Graph> bool lattice_constant_edge_equal(
                  typename boost::graph_traits<Subgraph>::edge_descriptor const & s_e
                , typename boost::graph_traits<Graph>::edge_descriptor const & g_e
                , Subgraph const & S
                , Graph const & G
                , boost::mpl::true_
            ) {
                return get(alps::edge_type_t(), S)[s_e] == get(alps::edge_type_t(), G)[g_e];
            }

            template<typename Subgraph, typename Graph> bool lattice_constant_vertex_equal(
                  typename boost::graph_traits<Subgraph>::vertex_descriptor const & s
                , typename boost::graph_traits<Graph>::vertex_descriptor const & g
                , Subgraph const & S
                , Graph const & G
                , boost::mpl::false_
            ) {
                return true;
            }

            template<typename Subgraph, typename Graph> bool lattice_constant_edge_equal(
                  typename boost::graph_traits<Subgraph>::edge_descriptor const & s_e
                , typename boost::graph_traits<Graph>::edge_descriptor const & g_e
                , Subgraph const & S
                , Graph const & G
                , boost::mpl::false_
            ) {
                return true;
            }

            // TODO: make an object out of walker
            template<typename Subgraph, typename Graph, typename ExitOnMatch> void lattice_constant_walker(
                  typename boost::graph_traits<Subgraph>::vertex_descriptor const & s
                , typename boost::graph_traits<Graph>::vertex_descriptor const & g
                , Subgraph const & S
                , Graph const & G
                , std::vector<std::size_t> const & I
#ifdef USE_COMPRESSED_EMBEDDING2
                , compressed_set<> & matches
#endif
#if defined(USE_COMPRESSED_EMBEDDING) || defined(CHECK_COMPRESSED_EMBEDDING)
                , boost::unordered_set<embedding_2d_type> & matches_2d
#endif
#if defined(USE_GENERIC_EMBEDDING) || defined(CHECK_COMPRESSED_EMBEDDING)
                , boost::unordered_set<embedding_generic_type> & matches_generic
#endif
                , std::vector<std::vector<boost::uint_t<8>::fast> > const & distance_to_boarder
                , std::deque<std::pair<
                      typename boost::graph_traits<Subgraph>::vertex_descriptor
                    , typename boost::graph_traits<Graph>::vertex_descriptor
                  > > stack
                , boost::dynamic_bitset<> placed
                , boost::dynamic_bitset<> & visited
                , std::vector<typename boost::graph_traits<Graph>::vertex_descriptor> & pinning
                , typename partition_type<Subgraph>::type const & subgraph_orbit
                , std::size_t unit_cell_size
                , ExitOnMatch exit_on_match
            ) {
                typedef typename boost::graph_traits<Subgraph>::vertex_descriptor SubgraphVertex;
                typedef typename boost::graph_traits<Graph>::vertex_descriptor GraphVertex;

                if (out_degree(s, S) > out_degree(g, G))
                    return;
                if (!lattice_constant_vertex_equal(
                      s
                    , g
                    , S
                    , G
                    , typename detail::has_coloring<typename boost::vertex_property_type<Graph>::type>::type())
                )
                    return;
                typename boost::graph_traits<Subgraph>::adjacency_iterator s_ai, s_ae;
                for (boost::tie(s_ai, s_ae) = adjacent_vertices(s, S); s_ai != s_ae; ++s_ai)
                    if (pinning[*s_ai] != num_vertices(G)) {
                        typename boost::graph_traits<Graph>::edge_descriptor e;
                        bool is_e;
                        boost::tie(e, is_e) = edge(g, pinning[*s_ai], G);
                        if (!is_e || !lattice_constant_edge_equal(
                              edge(s, *s_ai, S).first
                            , e
                            , S
                            , G
                            , typename detail::has_coloring<typename boost::edge_property_type<Graph>::type>::type())
                        )
                            return;
                    }
                visited[g] = true;
                pinning[s] = g;
                if (visited.count() < num_vertices(S)) {
                    typename boost::graph_traits<Graph>::adjacency_iterator g_ai, g_ae;
                    for (boost::tie(s_ai, s_ae) = adjacent_vertices(s, S); s_ai != s_ae; ++s_ai)
                        if (!placed[*s_ai]) {
                            placed[*s_ai] = true;
                            stack.push_back(std::make_pair(*s_ai, g));
                        }
                    SubgraphVertex t = stack[0].first;
                    boost::tie(g_ai, g_ae) = adjacent_vertices(stack[0].second, G);
                    stack.pop_front();
                    for (; g_ai != g_ae; ++g_ai)
                        if (!visited[*g_ai])
                            detail::lattice_constant_walker(
                                  t
                                , *g_ai
                                , S
                                , G
                                , I
#ifdef USE_COMPRESSED_EMBEDDING2
                                , matches
#endif                                
#if defined(USE_COMPRESSED_EMBEDDING) || defined(CHECK_COMPRESSED_EMBEDDING)
                                , matches_2d
#endif
#if defined(USE_GENERIC_EMBEDDING) || defined(CHECK_COMPRESSED_EMBEDDING)
                                , matches_generic
#endif
                                , distance_to_boarder
                                , stack
                                , placed
                                , visited
                                , pinning
                                , subgraph_orbit
                                , unit_cell_size
                                , exit_on_match
                            );
                } else
                    lattice_constant_insert<Subgraph, Graph, 20, 2>(
                          S
                        , G
                        , I
#ifdef USE_COMPRESSED_EMBEDDING2
                        , matches
#endif                                                        
#if defined(USE_COMPRESSED_EMBEDDING) || defined(CHECK_COMPRESSED_EMBEDDING)
                        , matches_2d
#endif
#if defined(USE_GENERIC_EMBEDDING) || defined(CHECK_COMPRESSED_EMBEDDING)
                        , matches_generic
#endif
                        , distance_to_boarder
                        , pinning
                        , subgraph_orbit
                        , unit_cell_size
                        , exit_on_match
                    );
                pinning[s] = num_vertices(G);
                visited[g] = false;
            }

            // Input: Subgraph, Graph, vertices of G contained in mapping of S on G
            // Output: lattice_constant of S in G containing v
            template<typename Subgraph, typename Graph, typename ExitOnMatch> std::size_t lattice_constant_impl(
                  Subgraph const & S
                , Graph const & G
                , std::vector<typename boost::graph_traits<Graph>::vertex_descriptor> const & V
                , std::vector<std::vector<boost::uint_t<8>::fast> > const & distance_to_boarder
                , typename partition_type<Subgraph>::type const & subgraph_orbit
                , std::size_t unit_cell_size
                , ExitOnMatch exit_on_match
            ) {
                // Assume the vertex desciptor is an unsigned integer type (since we want to use it as an index for a vector)
                BOOST_STATIC_ASSERT((boost::is_unsigned<typename alps::graph_traits<Subgraph>::vertex_descriptor>::value));
                assert(num_vertices(S) > 0);
                // if larger, extend the space
                assert(num_vertices(S) < 21);
                assert(num_edges(S) < 21);

                BOOST_STATIC_ASSERT((boost::is_unsigned<typename alps::graph_traits<Graph>::vertex_descriptor>::value));
                assert(num_vertices(G) > 0);
                
                // make sure, that a distance in one direction fits in a boost::uint8_t
                assert(num_vertices(G) < 256 * 256);

                // If the lattice has more than 2 dimensions improve lattice_constant_insert
                assert(distance_to_boarder.size() < 3);

                // orbit index => vertices
                std::vector<std::size_t> I(num_vertices(S));
                // Io = {(mi, j) : ni element of Vj
                for (typename partition_type<Subgraph>::type::const_iterator it = subgraph_orbit.begin(); it != subgraph_orbit.end(); ++it)
                    for (typename partition_type<Subgraph>::type::value_type::const_iterator jt = it->begin(); jt != it->end(); ++jt)
                        I[*jt] = it - subgraph_orbit.begin();

                // Matched embeddings
#ifdef USE_COMPRESSED_EMBEDDING2
                // TODO: use only 5 bits to save left offset -> 45 bits
                compressed_set<> matches(num_vertices(S) + 1);
#endif
#if defined(USE_COMPRESSED_EMBEDDING) || defined(CHECK_COMPRESSED_EMBEDDING)
                boost::unordered_set<embedding_2d_type> matches_2d;
#endif
#if defined(USE_GENERIC_EMBEDDING) || defined(CHECK_COMPRESSED_EMBEDDING)
                boost::unordered_set<embedding_generic_type> matches_generic;
#endif

                for (typename std::vector<typename boost::graph_traits<Graph>::vertex_descriptor>::const_iterator it = V.begin(); it != V.end(); ++it)
                    for (typename partition_type<Subgraph>::type::const_iterator jt = subgraph_orbit.begin(); jt != subgraph_orbit.end(); ++jt)
                        if (out_degree(jt->front(), S) <= out_degree(*it, G)) {
                            // TODO: use dynamicbitset
                            boost::dynamic_bitset<> placed(num_vertices(S));
                            boost::dynamic_bitset<> visited(num_vertices(G));
                            std::deque<std::pair<
                                  typename boost::graph_traits<Subgraph>::vertex_descriptor
                                , typename boost::graph_traits<Graph>::vertex_descriptor
                            > > stack;
                            std::vector<typename boost::graph_traits<Graph>::vertex_descriptor> pinning(num_vertices(S), num_vertices(G));
                            placed[jt->front()] = true;
                            lattice_constant_walker(
                                  jt->front()
                                , *it
                                , S
                                , G
                                , I 
#ifdef USE_COMPRESSED_EMBEDDING2
                                , matches
#endif
#if defined(USE_COMPRESSED_EMBEDDING) || defined(CHECK_COMPRESSED_EMBEDDING)
                                , matches_2d
#endif
#if defined(USE_GENERIC_EMBEDDING) || defined(CHECK_COMPRESSED_EMBEDDING)
                                , matches_generic
#endif
                                , distance_to_boarder
                                , stack
                                , placed
                                , visited
                                , pinning
                                , subgraph_orbit
                                , unit_cell_size
                                , exit_on_match
                            );
                            break;
                        }
#ifdef USE_COMPRESSED_EMBEDDING2
//                std::cout << "average walk: " << matches.avgwalk / double(matches.size()) << std::endl;
                return matches.size();
#else
#if defined(USE_COMPRESSED_EMBEDDING) || defined(CHECK_COMPRESSED_EMBEDDING)
                return matches_2d.size();
#else
                return matches_generic.size();
#endif
#endif
            }
        }

        template<typename Subgraph, typename Graph, typename Lattice> std::size_t lattice_constant(
              Subgraph const & S
            , Graph const & G
            , Lattice const & L
            , typename alps::lattice_traits<Lattice>::cell_descriptor c
        ) {            
            typedef typename alps::graph_helper<Graph>::lattice_type lattice_type;
            typedef typename alps::lattice_traits<lattice_type>::unit_cell_type::graph_type unit_cell_graph_type;
            typedef typename boost::graph_traits<Graph>::vertex_descriptor vertex_descriptor;
            typedef typename alps::lattice_traits<Lattice>::size_type cell_index_type;

            // Get the possible translation in the lattice
            std::vector<std::vector<boost::uint_t<8>::fast> > distance_to_boarder(dimension(L), std::vector<boost::uint_t<8>::fast>(num_vertices(G), num_vertices(G)));
            detail::build_translation_table(G, L, distance_to_boarder);

            typename partition_type<Subgraph>::type subgraph_orbit = boost::get<2>(canonical_properties(S));

            const cell_index_type cell_id = index(c, L);
            std::size_t unit_cell_size = num_vertices(alps::graph::graph(unit_cell(L)));
            std::vector<typename boost::graph_traits<Graph>::vertex_descriptor> V;
            for(unsigned v = 0; v < unit_cell_size; ++v)
                V.push_back(cell_id * unit_cell_size + v);

            return detail::lattice_constant_impl(S, G, V, distance_to_boarder, subgraph_orbit, unit_cell_size, boost::mpl::false_());
        }

        template<typename Subgraph, typename Graph> bool is_embeddable(
              Subgraph const & S
            , Graph const & G
            , typename boost::graph_traits<Graph>::vertex_descriptor v
            , typename partition_type<Subgraph>::type const & subgraph_orbit            
        ) {
            std::vector<std::vector<boost::uint_t<8>::fast> > distance_to_boarder;

            try {
                std::vector<typename boost::graph_traits<Graph>::vertex_descriptor> V(1, v);
                detail::lattice_constant_impl(S, G, V, distance_to_boarder, subgraph_orbit, 1, boost::mpl::true_());
                return false;
            } catch (detail::embedding_found e) {
                return true;
            }
        }

        template<typename Subgraph, typename Graph> bool is_embeddable(
              Subgraph const & S
            , Graph const & G
            , typename partition_type<Subgraph>::type const & subgraph_orbit
        ) {
            std::vector<std::vector<boost::uint_t<8>::fast> > distance_to_boarder;

            try {
                typename boost::graph_traits<Graph>::vertex_iterator vt, ve;
                for (boost::tie(vt, ve) = vertices(G); vt != ve; ++vt) {
                    std::vector<typename boost::graph_traits<Graph>::vertex_descriptor> V(1, *vt);
                    detail::lattice_constant_impl(S, G, V, distance_to_boarder, subgraph_orbit, 1, boost::mpl::true_());
                }
                return false;
            } catch (detail::embedding_found e) {
                return true;
            }
        }        
    }
}
#endif
