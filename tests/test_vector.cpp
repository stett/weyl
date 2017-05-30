#include <cmath>
#include "catch.hpp"
#include "vector.h"
using namespace weyl;

TEST_CASE("Vector dot product", "[vector]") {
    vector<float, 3> v1(1.0f, 2.0f, 3.0f);
    vector<float, 3> v2(4.0f, 5.0f, 6.0f);
    float product = dot(v1, v2);
    float expected_product = (1.0f * 4.0f) + (2.0f * 5.0f) + (3.0f * 6.0f);
    REQUIRE(product == expected_product);
}

TEST_CASE("Vector cross product (3D)", "[vector]") {
    vector<float, 3> v1(1.0f, 2.0f, 3.0f);
    vector<float, 3> v2(4.0f, 5.0f, 6.0f);
    vector<float, 3> product = cross(v1, v2);
    vector<float, 3> expected_product(-3.0f, 6.0f, -3.0f);
    REQUIRE(product == expected_product);
}

TEST_CASE("Vector cross product (mock - 2D)", "[vector]") {
    vector<float, 2> v1(1.0f, 2.0f);
    vector<float, 2> v2(3.0f, 4.0f);
    float product = cross(v1, v2);
    float expected_product = 4.0f - 6.0f;
    REQUIRE(product == expected_product);
}

TEST_CASE("Vector magnitude squared", "[vector]") {
    vector<float, 2> v(1.0f, 2.0f);
    REQUIRE(magnitude_sq(v) == 5.0f);
}

TEST_CASE("Vector magnitude", "[vector]") {
    vector<float, 2> v(1.0f, 2.0f);
    REQUIRE(magnitude(v) == std::sqrt(5.0f));
}
