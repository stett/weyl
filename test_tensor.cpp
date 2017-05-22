#include "catch.hpp"
#include "tensor.h"
using namespace weyl;

TEST_CASE("Tensor default constructor", "[tensor]") {
    tensor<float, 2, 3, 1> t;
}

TEST_CASE("Tensor single-value constructor", "[tensor]") {
    const float value = 4.0f;
    tensor<float, 2, 3, 1> t(value);
    for (size_t i = 0; i < 2; ++i)
    for (size_t j = 0; j < 3; ++j)
    for (size_t k = 0; k < 1; ++k)
        REQUIRE(t[i][j][k] == value);
}

TEST_CASE("Tensor initializer-list constructor", "[tensor]") {
    tensor<float, 2, 3> t({ { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } });
    REQUIRE(t[0][0] == 1.0f);
    REQUIRE(t[0][1] == 2.0f);
    REQUIRE(t[0][2] == 3.0f);
    REQUIRE(t[1][0] == 4.0f);
    REQUIRE(t[1][1] == 5.0f);
    REQUIRE(t[1][2] == 6.0f);
}

TEST_CASE("Single-index tensor variadic constructor", "[tensor]") {
    tensor<float, 3> t(1.0f, 2.0f, 3.0f);
    REQUIRE(t[0] == 1.0f);
    REQUIRE(t[1] == 2.0f);
    REQUIRE(t[2] == 3.0f);
}

TEST_CASE("Copy constructor", "[tensor]") {
    tensor<float, 2, 3> t1({ { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } });
    tensor<float, 2, 3> t2(t1);
    REQUIRE(t2[0][0] == 1.0f);
    REQUIRE(t2[0][1] == 2.0f);
    REQUIRE(t2[0][2] == 3.0f);
    REQUIRE(t2[1][0] == 4.0f);
    REQUIRE(t2[1][1] == 5.0f);
    REQUIRE(t2[1][2] == 6.0f);
}

TEST_CASE("Tensor dimensionality internals", "[tensor]") {
    REQUIRE((detail::Dimension<0, 1, 2, 3>::value == 1));
    REQUIRE((detail::Dimension<1, 1, 2, 3>::value == 2));
    REQUIRE((detail::Dimension<2, 1, 2, 3>::value == 3));
}

TEST_CASE("Single index reduction internals (3D)", "[tensor]") {
    using ReducedT = weyl::detail::Reduction<0, 1, float>::Post<1, 2, 3>::reduced_tensor_t;
    REQUIRE(ReducedT::rank::value == 2);
    REQUIRE(ReducedT::dimension<0>::value == 1);
    REQUIRE(ReducedT::dimension<1>::value == 3);
}

TEST_CASE("Single index reduction internals (9D)", "[tensor]") {
    using ReducedT = weyl::detail::Reduction<0, 6, float>::Post<1, 2, 3, 4, 5, 6, 7, 8, 9>::reduced_tensor_t;
    REQUIRE(ReducedT::rank::value == 8);
    REQUIRE(ReducedT::dimension<0>::value == 1);
    REQUIRE(ReducedT::dimension<1>::value == 2);
    REQUIRE(ReducedT::dimension<2>::value == 3);
    REQUIRE(ReducedT::dimension<3>::value == 4);
    REQUIRE(ReducedT::dimension<4>::value == 5);
    REQUIRE(ReducedT::dimension<5>::value == 6);
    REQUIRE(ReducedT::dimension<6>::value == 8);
    REQUIRE(ReducedT::dimension<7>::value == 9);
}

//TEST_CASE("Double index reduction internals (9D)", "[tensor]") {
//    using ReducedT = weyl::detail::Reduction<3, 6, float>::Post<1, 2, 3, 4, 5, 6, 7, 8, 9>::reduced_tensor_t;
    /*
    REQUIRE(ReducedT::rank::value == 7);
    REQUIRE(ReducedT::dimension<0>::value == 1);
    REQUIRE(ReducedT::dimension<1>::value == 2);
    REQUIRE(ReducedT::dimension<2>::value == 3);
    REQUIRE(ReducedT::dimension<3>::value == 5);
    REQUIRE(ReducedT::dimension<4>::value == 6);
    REQUIRE(ReducedT::dimension<5>::value == 8);
    REQUIRE(ReducedT::dimension<6>::value == 9);
    */
//}

TEST_CASE("Tensor rank", "[tensor]") {
    using T = tensor<float, 2, 3, 1>;
    REQUIRE(T::rank::value == 3);
}

TEST_CASE("Tensor dimensionality", "[tensor]") {
    using T = tensor<float, 2, 3, 1>;
    REQUIRE(T::dimension<0>::value == 2);
    REQUIRE(T::dimension<1>::value == 3);
    REQUIRE(T::dimension<2>::value == 1);
}

TEST_CASE("Correct tensor summations are callable", "[tensor]") {
    tensor<float, 2, 3, 1> t1;
    tensor<float, 4, 2, 3> t2;
    t1.sum<0, 1>(t2);
    t1.sum<1, 2>(t2);
    t2.sum<1, 0>(t1);
    t2.sum<2, 1>(t1);
}

TEST_CASE("Tensor comparison", "[tensor]") {
    tensor<float, 2, 3> t1({ { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } });
    tensor<float, 2, 3> t2({ { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } });
    tensor<float, 2, 3> t3({ { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 0.0f } });
    tensor<float, 3, 2> t4({ { 1.0f, 2.0f }, { 3.0f, 4.0f }, { 5.0f, 6.0f } });
    REQUIRE(t1 == t2);
    REQUIRE(t1 != t3);
    REQUIRE(t1 != t4);
    REQUIRE(!(t1 != t2));
    REQUIRE(!(t1 == t3));
    REQUIRE(!(t1 == t4));
}

TEST_CASE("Single index tensor summation", "[tensor]") {
    using T = tensor<float, 3>;
    T t1(1.0f, 2.0f, 3.0f);
    T t2(2.0f, 3.0f, 4.0f);
    float sum = t1.sum<0, 0>(t2);
    float expected_sum = (1.0f * 2.0f) + (2.0f * 3.0f) + (3.0f * 4.0f);
    REQUIRE(sum == expected_sum);
}

/*
TEST_CASE("2x2 * 2x2 tensor product", "[tensor]") {
    tensor<float, 2, 2> t1({
        { 1.0f, 2.0f },
        { 3.0f, 4.0f }
    });

    tensor<float, 2, 2> t2({
        { 1.0f, 2.0f },
        { 3.0f, 4.0f }
    });

    tensor<float, 2, 2> expected_sum({
        { (1.0f * 1.0f) + (2.0f * 3.0f), (1.0f * 2.0f) + (2.0f * 4.0f) },
        { (1.0f * 3.0f) + (3.0f * 4.0f), (2.0f * 3.0f) + (4.0f * 4.0f) }
    });

    tensor<float, 2, 2> sum = t1.sum<1, 0>(t2);
    bool equal = sum == expected_sum;
    REQUIRE(equal);
}
*/

TEST_CASE("Multiple index tensor summation", "[tensor]") {

    tensor<float, 2, 3, 2> t1({
        { { 1.0f, 2.0f }, { 2.0f, 3.0f }, { 3.0f, 4.0f } },
        { { 4.0f, 5.0f }, { 5.0f, 6.0f }, { 6.0f, 7.0f } }
    });

    tensor<float, 3, 2, 2> t2({
        { { 1.0f, 2.0f }, { 2.0f, 3.0f } },
        { { 2.0f, 3.0f }, { 3.0f, 4.0f } },
        { { 3.0f, 4.0f }, { 4.0f, 5.0f } }
    });

    tensor<float, 3, 2> expected_sum({
        { 1.0f * 1.0f + 2.0f * 2.0f }, // { 1.0f, 2.0f }, { 2.0f, 3.0f }, { 3.0f, 4.0f } * { 1.0f, 2.0f }, { 2.0f, 3.0f }, { 3.0f, 4.0f },
        { 2.0f * 2.0f + 3.0f * 3.0f },
        { 3.0f * 3.0f + 4.0f * 4.0f }
    });

    //REQUIRE(t1.sum<>);
}
