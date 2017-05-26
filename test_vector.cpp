#include <type_traits>
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
    vector<int, 3> v1(1, 2, 3);
    vector<int, 3> v2(4, 5, 6);
    vector<int, 3> product = cross(v1, v2);
    vector<int, 3> expected_product(-3, 6, -3);
    REQUIRE(product == expected_product);
}

TEST_CASE("Vector cross product (mock - 2D)", "[vector]") {
    vector<float, 2> v1(1.0f, 2.0f);
    vector<float, 2> v2(3.0f, 4.0f);
    float product = cross(v1, v2);
    float expected_product = 4.0f - 6.0f;
    REQUIRE(product == expected_product);
}
