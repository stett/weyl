#include <cmath>
#include "catch.hpp"
using namespace weyl;

TEST_CASE("Vector dot product", "[vector]") {
    tensor<float, 3> v1(1.0f, 2.0f, 3.0f);
    tensor<float, 3> v2(4.0f, 5.0f, 6.0f);
    float product = inner(v1, v2);
    float expected_product = (1.0f * 4.0f) + (2.0f * 5.0f) + (3.0f * 6.0f);
    REQUIRE(product == expected_product);
}

TEST_CASE("Vector cross product (3D)", "[vector]") {
    tensor<float, 3> v1(1.0f, 2.0f, 3.0f);
    tensor<float, 3> v2(4.0f, 5.0f, 6.0f);
    tensor<float, 3> product = cross(v1, v2);
    tensor<float, 3> expected_product(-3.0f, 6.0f, -3.0f);
    REQUIRE(product == expected_product);
}

TEST_CASE("Vector cross product (mock - 2D)", "[vector]") {
    tensor<float, 2> v1(1.0f, 2.0f);
    tensor<float, 2> v2(3.0f, 4.0f);
    float product = cross(v1, v2);
    float expected_product = 4.0f - 6.0f;
    REQUIRE(product == expected_product);
}

TEST_CASE("Vector magnitude squared", "[vector]") {
    tensor<float, 2> v(1.0f, 2.0f);
    REQUIRE(magnitude_sq(v) == 5.0f);
}

TEST_CASE("Vector magnitude", "[vector]") {
    tensor<float, 2> v(1.0f, 2.0f);
    REQUIRE(magnitude(v) == std::sqrt(5.0f));
}
