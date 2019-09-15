#include <cmath>
#include "catch.hpp"
#include "weyl.h"
using namespace weyl;

TEST_CASE("Vector dot product", "[vector]") {
    vector<float, 3> v1(1.0f, 2.0f, 3.0f);
    vector<float, 3> v2(4.0f, 5.0f, 6.0f);
    float product = inner(v1, v2);
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
    REQUIRE(v.magnitude_sq() == 5.0f);
}

TEST_CASE("Vector magnitude", "[vector]") {
    vector<float, 2> v(1.0f, 2.0f);
    REQUIRE(v.magnitude() == std::sqrt(5.0f));
}

TEST_CASE("Vector member access", "[vector]") {
    vector<float, 2> v2(1.0f, 2.0f);
    vector<float, 3> v3(1.0f, 2.0f, 3.0f);
    vector<float, 4> v4(1.0f, 2.0f, 3.0f, 4.0f);
    REQUIRE(v2.x == 1.0f);
    REQUIRE(v2.y == 2.0f);
    REQUIRE(v3.x == 1.0f);
    REQUIRE(v3.y == 2.0f);
    REQUIRE(v3.z == 3.0f);
    REQUIRE(v4.x == 1.0f);
    REQUIRE(v4.y == 2.0f);
    REQUIRE(v4.z == 3.0f);
    REQUIRE(v4.w == 4.0f);
}

TEST_CASE("Vector/subvector initialization", "[vector]") {
    vector<float, 2> v2(1.f, 2.f);
    vector<float, 3> v3(v2, 3.f);
    vector<float, 5> v5(v3, 4.f, 5.f);
    REQUIRE(v5[0] == 1.f);
    REQUIRE(v5[1] == 2.f);
    REQUIRE(v5[2] == 3.f);
    REQUIRE(v5[3] == 4.f);
    REQUIRE(v5[4] == 5.f);
}
