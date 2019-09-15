#include "catch.hpp"
#include "weyl.h"
using namespace weyl;

#include <iostream>

TEST_CASE("Matrix product (square)", "[matrix]") {

    matrix<float, 3, 3> m1({
        { 1.0f, 2.0f, 3.0f },
        { 2.0f, 3.0f, 4.0f },
        { 3.0f, 4.0f, 5.0f }
    });

    matrix<float, 3, 3> m2({
        { 4.0f, 5.0f, 6.0f },
        { 5.0f, 6.0f, 7.0f },
        { 6.0f, 7.0f, 8.0f }
    });

    matrix<float, 3, 3> product = m1 * m2;

    matrix<float, 3, 3> expected_product({
        { 32.0f, 38.0f, 44.0f },
        { 47.0f, 56.0f, 65.0f },
        { 62.0f, 74.0f, 86.0f }
    });

    REQUIRE(product == expected_product);
}

TEST_CASE("Matrix row extraction", "[matrix]") {
    matrix<float, 3, 3> m({
        { 1.0f, 2.0f, 3.0f },
        { 4.0f, 5.0f, 6.0f },
        { 7.0f, 8.0f, 9.0f }
    });

    REQUIRE((m.row<0>() == vector<float, 3>(1.0f, 2.0f, 3.0f)));
    REQUIRE((m.row<1>() == vector<float, 3>(4.0f, 5.0f, 6.0f)));
    REQUIRE((m.row<2>() == vector<float, 3>(7.0f, 8.0f, 9.0f)));
    REQUIRE((m.row(0) == vector<float, 3>(1.0f, 2.0f, 3.0f)));
    REQUIRE((m.row(1) == vector<float, 3>(4.0f, 5.0f, 6.0f)));
    REQUIRE((m.row(2) == vector<float, 3>(7.0f, 8.0f, 9.0f)));
}

TEST_CASE("Matrix column extraction", "[matrix]") {
    matrix<float, 3, 3> m({
        { 1.0f, 2.0f, 3.0f },
        { 4.0f, 5.0f, 6.0f },
        { 7.0f, 8.0f, 9.0f }
    });

    REQUIRE((m.col<0>() == vector<float, 3>(1.0f, 4.0f, 7.0f)));
    REQUIRE((m.col<1>() == vector<float, 3>(2.0f, 5.0f, 8.0f)));
    REQUIRE((m.col<2>() == vector<float, 3>(3.0f, 6.0f, 9.0f)));
    REQUIRE((m.col(0) == vector<float, 3>(1.0f, 4.0f, 7.0f)));
    REQUIRE((m.col(1) == vector<float, 3>(2.0f, 5.0f, 8.0f)));
    REQUIRE((m.col(2) == vector<float, 3>(3.0f, 6.0f, 9.0f)));
}

TEST_CASE("Matrix inverse", "[matrix]") {
    matrix<float, 2, 2> m({
        {1, 2},
        {3, 4},
    });
    matrix<float, 2, 2> m_inv = m.inverse();
    matrix<float, 2, 2> m_inv_expected({
        {-2, 1},
        {3.f/2.f, -.5f},
    });
    REQUIRE(m_inv == m_inv_expected);
}
