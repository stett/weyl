#include "catch.hpp"
#include "weyl.h"
using namespace weyl;

#include <iostream>

TEST_CASE("2x2 Matrix operations", "[Matrix]") {

    mat<2, 2> A({ { 1.0f, 2.0f }, { 3.0f, 4.0f } });

    SECTION("Determinant, 2x2") {
        float A_det_expected = -2.0f;
        float A_det = A.det();
        REQUIRE(A_det_expected == A_det);
    }

    SECTION("Transpose, 2x2") {
        mat<2, 2> A_transpose_expected({ { 1.0f, 3.0f }, { 2.0f, 4.0f } });
        mat<2, 2> A_transpose = A.transpose();
        REQUIRE(A_transpose == A_transpose_expected);
    }

    SECTION("Minor, 2x2") {
        REQUIRE((A.minor<0,0>() == 4.f));
        REQUIRE((A.minor<0,1>() == 3.f));
        REQUIRE((A.minor<1,0>() == 2.f));
        REQUIRE((A.minor<1,1>() == 1.f));
    }

    SECTION("Minors, 2x2") {
        mat<2, 2> A_minors_expected({ { 4.0f, 3.0f }, { 2.0f, 1.0f } });
        mat<2, 2> A_minors = A.minors();
        REQUIRE(A_minors == A_minors_expected);
    }

    SECTION("Cofactor, 2x2") {
        mat<2, 2> A_cofactor_expected({ { 4.0f, -3.0f }, { -2.0f, 1.0f } });
        mat<2, 2> A_cofactor = A.cofactor();
        REQUIRE(A_cofactor == A_cofactor_expected);
    }

    SECTION("adjugate, 2x2") {
        mat<2, 2> A_adj_expected({ { 4.0f, -2.0f }, { -3.0f, 1.0f } });
        mat<2, 2> A_adj = A.adj();
        REQUIRE(A_adj == A_adj_expected);
    }

    SECTION("Matrix inversion, 2x2") {
        mat<2, 2> A_inv_expected({ { -2.0f, 1.0f }, { 3.0f/2.0f, -1.0f/2.0f } });
        mat<2, 2> A_inv = A.inverse();
        REQUIRE(A_inv == A_inv_expected);
    }
}

TEST_CASE("3x3 Matrix operations", "[Matrix]") {

    mat<3, 3> A({
        { -1.0f, 2.0f, 3.0f },
        {  2.0f, 3.0f, 4.0f },
        { -3.0f, 4.0f, 5.0f } });

    SECTION("Element access") {
        REQUIRE(A[0][0] == -1.f);
        REQUIRE(A[0][1] == 2.f);
        REQUIRE(A[0][2] == 3.f);
        REQUIRE(A[1][0] == 2.f);
        REQUIRE(A[1][1] == 3.f);
        REQUIRE(A[1][2] == 4.f);
        REQUIRE(A[2][0] == -3.f);
        REQUIRE(A[2][1] == 4.f);
        REQUIRE(A[2][2] == 5.f);
    }

    SECTION("Minor 0,0") {
        mat<2, 2> A_minor_00({ { 3.0f, 4.0f }, { 4.0f, 5.0f } });
        REQUIRE((A.minor<0, 0>() == A_minor_00));
    }

    SECTION("Minor 1,0") {
        mat<2, 2> A_minor_10({ { 2.0f, 3.0f }, { 4.0f, 5.0f } });
        REQUIRE((A.minor<1, 0>() == A_minor_10));
    }

    SECTION("Minor 0,2") {
        mat<2, 2> A_minor_02({ { 2.0f, 3.0f }, { -3.0f, 4.0f } });
        REQUIRE((A.minor<0, 2>() == A_minor_02));
    }

    /*
    SECTION("Determinant, 3x3") {
        float A_det_expected = 8.0f;
        float A_det = A.det();
        REQUIRE(A_det_expected == A_det);
    }

    SECTION("Transpose, 3x3") {
        mat<3, 3> A_transpose_expected({ { -1.0f, 2.0f, -3.0f }, { 2.0f, 3.0f, 4.0f }, { 3.0f, 4.0f, 5.0f } });
        mat<3, 3> A_transpose = A.transpose();
        REQUIRE(A_transpose == A_transpose_expected);
    }

    SECTION("Minors, 3x3") {
        mat<3, 3> A_minors_expected({ { -1.0f, -2.0f, -1.0f }, { 22.0f, 4.0f, -10.0f }, { 17.0f, 2.0f, -7.0f } });
        mat<3, 3> A_minors = A.minors();
        REQUIRE(A_minors == A_minors_expected);
    }

    SECTION("Cofactor, 3x3") {
        mat<3, 3> A_cofactor_expected({ { -1.0f, -22.0f, 17.0f }, { 2.0f, 4.0f, -2.0f }, { -1.0f, 10.0f, -7.0f } });
        mat<3, 3> A_cofactor = A.cofactor();
        REQUIRE(A_cofactor == A_cofactor_expected);
    }

    SECTION("Adjugate, 3x3") {
        mat<3, 3> A_adjugate_expected({ { -1.0f, 2.0f, -1.0f }, { -22.0f, 4.0f, 10.0f }, { 17.0f, -2.0f, -7.0f } });
        mat<3, 3> A_adjugate = A.adj();
        REQUIRE(A_adjugate == A_adjugate_expected);
    }

    SECTION("Inverse, 3x3") {
        mat<3, 3> A_inverse_expected({
            { -1.0f/8.0f, 1.0f/4.0f, -1.0f/8.0f },
            { -11.0f/4.0f, 1.0f/2.0f, 5.0f/4.0f },
            { 17.0f/8.0f, -1.0f/4.0f, -7.0f/8.0f } });
        mat<3, 3> A_inverse = A.inverse();
        mat<3, 3> A_inverse_A = A.inverse() * A;
        mat<3, 3> A_A_inverse = A * A.inverse();
        REQUIRE(A_inverse == A_inverse_expected);
        REQUIRE((A_inverse_A == mat<3, 3>()));
        REQUIRE((A_A_inverse == mat<3, 3>()));
    }

    SECTION("Columns, 3x3") {
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

    SECTION("Rows, 3x3") {
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

    SECTION("Product, 3x3") {
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
    */
}
