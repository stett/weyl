#include "catch.hpp"
#include "weyl.h"
using namespace weyl;


TEST_CASE("Matrix product (square)", "[matrix]") {

    tensor<float, 3, 3> m1({
        { 1.0f, 2.0f, 3.0f },
        { 2.0f, 3.0f, 4.0f },
        { 3.0f, 4.0f, 5.0f }
    });

    tensor<float, 3, 3> m2({
        { 4.0f, 5.0f, 6.0f },
        { 5.0f, 6.0f, 7.0f },
        { 6.0f, 7.0f, 8.0f }
    });

    tensor<float, 3, 3> product = weyl::product(m1, m2);

    tensor<float, 3, 3> expected_product({
        { 32.0f, 38.0f, 44.0f },
        { 47.0f, 56.0f, 65.0f },
        { 62.0f, 74.0f, 86.0f }
    });

    REQUIRE(product == expected_product);
}

TEST_CASE("Matrix row extraction", "[matrix]") {
    tensor<float, 3, 3> m({
        { 1.0f, 2.0f, 3.0f },
        { 4.0f, 5.0f, 6.0f },
        { 7.0f, 8.0f, 9.0f }
    });

    REQUIRE((row<0>(m) == tensor<float, 3>(1.0f, 2.0f, 3.0f)));
    REQUIRE((row<1>(m) == tensor<float, 3>(4.0f, 5.0f, 6.0f)));
    REQUIRE((row<2>(m) == tensor<float, 3>(7.0f, 8.0f, 9.0f)));
    REQUIRE((row(m, 0) == tensor<float, 3>(1.0f, 2.0f, 3.0f)));
    REQUIRE((row(m, 1) == tensor<float, 3>(4.0f, 5.0f, 6.0f)));
    REQUIRE((row(m, 2) == tensor<float, 3>(7.0f, 8.0f, 9.0f)));
}

TEST_CASE("Matrix column extraction", "[matrix]") {
    tensor<float, 3, 3> m({
        { 1.0f, 2.0f, 3.0f },
        { 4.0f, 5.0f, 6.0f },
        { 7.0f, 8.0f, 9.0f }
    });

    REQUIRE((col<0>(m) == tensor<float, 3>(1.0f, 4.0f, 7.0f)));
    REQUIRE((col<1>(m) == tensor<float, 3>(2.0f, 5.0f, 8.0f)));
    REQUIRE((col<2>(m) == tensor<float, 3>(3.0f, 6.0f, 9.0f)));
    REQUIRE((col(m, 0) == tensor<float, 3>(1.0f, 4.0f, 7.0f)));
    REQUIRE((col(m, 1) == tensor<float, 3>(2.0f, 5.0f, 8.0f)));
    REQUIRE((col(m, 2) == tensor<float, 3>(3.0f, 6.0f, 9.0f)));
}