#include <iostream>
#include <chrono>
#include <boost/math/special_functions/binomial.hpp>
#include <boost/json.hpp>
#include <boost/json/object.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <omp.h>

#define TOLERANCE 1E-5

namespace bm = boost::math;
namespace b = boost;
namespace json = boost::json;

using diceInt = boost::multiprecision::cpp_bin_float_oct;

json::object run_simulation(double dicesides, double dicecount);

/* https://stackoverflow.com/questions/22685563/integer-calculation-of-the-binomial-coefficient-using-boostmathbinomial-coef */
diceInt BinomialCoefficient(const diceInt& n, const diceInt& k) {
    if (k == 0) { return 1; }
    else { return (n * BinomialCoefficient(n - 1, k - 1)) / k; }
}

diceInt chance(const diceInt& points, const diceInt& dice, const diceInt& sides) {
    const diceInt lim = floor((points - dice) / sides);

    std::vector<diceInt> sumVec((int)lim + 1);

    #pragma omp parallel for
    for (int i = 0; i <= (int)lim; ++i) {
        const auto sign = pow(-1, i);
        const auto c2 = BinomialCoefficient(points - sides * i - 1, dice - 1);
        const auto c1 = BinomialCoefficient(dice, i);
        sumVec[i] = sign * c1 * c2;
    }

    diceInt sum = 0;
    for (int i = 0; i <= (int)lim; ++i) {
        sum += sumVec[i];
    }

    return diceInt (sum) / diceInt (pow(sides, dice));
}


int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "[dice count] [dice sides]" << std::endl;
        return 0;
    }

    auto dicesides = b::lexical_cast<double>(argv[2]);
    auto dicecount = b::lexical_cast<double>(argv[1]);

    bool single_sim = false;

    if (argc > 2)
        single_sim = true;

    std::cerr << "num type size: " << sizeof (diceInt) << std::endl;

    json::object output;

    auto start = std::chrono::high_resolution_clock::now();

    if (!single_sim) {

#pragma omp parallel
        {
#pragma omp single
            {
                for (int i = 1; i <= dicecount; i++) {
#pragma omp task firstprivate(i)
                    {
                        char line[1024];
                        snprintf(line, sizeof line, "simulating %dd%.0f (thread: %d)\n", i, dicesides,
                                 omp_get_thread_num());
                        std::cerr << line;
                        auto item = run_simulation(dicesides, i);
#pragma omp critical
                        output[b::lexical_cast<std::string>(i)] = item;
                    }
                }
            }
        }
    } else {
        auto item = run_simulation(dicesides, dicecount);
        output[b::lexical_cast<std::string>(dicecount)] = item;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration<double, std::ratio<1, 1000>>(end - start).count();
    std::cerr << "all done in " << ms << "ms.\n";

    std::cout << output;
    return 0;
}

json::object run_simulation(double dicesides, double dicecount) {
    const auto minroll = (int)dicecount;
    const auto maxroll = (int) (dicesides * dicecount);
    std::vector<diceInt> probVec(maxroll - minroll + 1);

    // #pragma omp parallel for
    for (int l = minroll; l <= maxroll; l++) {
        const auto prob = chance(l, dicecount, dicesides) * 100;
        probVec[l - minroll] = prob;

        if (prob < 0 || prob > 100) {
            diceInt clampVal;

            if (prob < 0)
                clampVal = - prob;
            else
                clampVal = prob - 100;

            if (clampVal > TOLERANCE)
                std::cerr << "bad probability at xdx = " << l << " of " << prob << "%. Error is " << clampVal << "\n";
        }
    }

    json::object object;

    diceInt sum = 0;
    for (int l = minroll; l <= maxroll; ++l) {
        auto prob = probVec[l - minroll];

        sum += prob;
        json::object item = {
                {"prob", (double) prob},
                {"sum",  (double) sum}
        };

        object[b::lexical_cast<std::string>(l)] = item;
    }

    return object;
}
