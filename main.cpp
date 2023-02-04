#include <iostream>
#include <chrono>
#include <boost/math/special_functions/binomial.hpp>
#include <boost/json.hpp>
#include <boost/json/object.hpp>
#include <omp.h>

namespace bm = boost::math;
namespace b = boost;
namespace json = boost::json;

using diceInt = double;

json::object run_simulation(double dicesides, double dicecount);

double chance(diceInt points, diceInt dice, diceInt sides) {
    diceInt sum = 0;
    const diceInt lim = floor((points - dice) / sides);

    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i <= (int)lim; ++i) {
        auto sign = pow(-1, i);
        auto c2 = bm::binomial_coefficient<diceInt>(points - sides * i - 1, dice - 1);
        auto c1 = bm::binomial_coefficient<diceInt>(dice, i);
        sum += sign * c1 * c2;
    }

    return double (sum) / double (pow(sides, dice));
}


int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "[dice count] [dice sides]" << std::endl;
        return 0;
    }

    auto dicesides = b::lexical_cast<double>(argv[2]);
    auto dicecount = b::lexical_cast<double>(argv[1]);

    json::object output;

    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int i = 1; i <= dicecount; i++) {
                #pragma omp task firstprivate(i)
                {
                    char line[1024];
                    snprintf(line, sizeof line, "simulating %dd%.0f (thread: %d)\n", i, dicesides, omp_get_thread_num());
                    std::cerr << line;
                    auto item = run_simulation(dicesides, i);
                    #pragma omp critical
                    output[b::lexical_cast<std::string>(i)] = item;
                }
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration<double, std::ratio<1, 1000>>(end - start).count();
    std::cerr << "all done in " << ms << "ms.\n";

    std::cout << output;
    return 0;
}

json::object run_simulation(double dicesides, double dicecount) {
    auto minroll = dicecount;
    auto maxroll = dicesides * dicecount;

    json::object object;
    std::cout.precision(std::numeric_limits<double>::max_digits10);

    double sum = 0;
    for (int l = minroll; l <= maxroll; ++l) {
        auto prob = chance(l, dicecount, dicesides) * 100;
        sum += prob;

        json::object item = {
                {"prob", prob},
                {"sum", sum}
        };

        object[b::lexical_cast<std::string>(l)] = item;
    }

    return object;
}
