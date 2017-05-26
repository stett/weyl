#pragma once
#include "tensor.h"
#include "vector.h"

namespace weyl
{
    template <typename T, size_t Rows, size_t Cols>
    class matrix : public tensor<T, Rows, Cols>
    {
    public:

        matrix(const tensor<T, Rows, Cols>& tens) : tensor<T, Rows, Cols>(tens) {}

        matrix(const std::initializer_list< std::initializer_list< T > >& initial) : tensor<T, Rows, Cols>(initial) {}

        /// \brief Matrix product
        template <size_t OtherCols>
        typename tensor<T, Rows, Cols>::template convolution_t<1, 0, Cols, OtherCols>
        operator*(const matrix<T, Cols, OtherCols>& other) {
            return sum<1, 0>(*this, other);
        }
    };
}