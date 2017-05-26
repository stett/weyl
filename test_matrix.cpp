#include "catch.hpp"
#include "matrix.h"
using namespace weyl;


TEST_CASE("Matrix product (square)", "[matrix]") {
    matrix<float, 3, 3> a({
        { 1.0f, 2.0f, 3.0f },
        { 2.0f, 3.0f, 4.0f },
        { 3.0f, 4.0f, 5.0f }
    });

    matrix<float, 3, 3> b({
        { 4.0f, 5.0f, 6.0f },
        { 5.0f, 6.0f, 7.0f },
        { 6.0f, 7.0f, 8.0f }
    });

    matrix<float, 3, 3> product = a * b;

    matrix<float, 3, 3> expected_product({
        { 32.0f, 38.0f, 44.0f },
        { 47.0f, 56.0f, 65.0f },
        { 62.0f, 74.0f, 86.0f }
    });

    REQUIRE(product == expected_product);
}