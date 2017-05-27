#pragma once
#include "tensor.h"
#include "vector.h"

namespace weyl
{
    template <typename T, size_t Rows, size_t Cols>
    class matrix : public tensor<T, Rows, Cols>
    {
    public:

        using tensor_t = tensor<T, Rows, Cols>;

        matrix(const tensor<T, Rows, Cols>& tens) : tensor_t(tens) {}

        matrix(const std::initializer_list< std::initializer_list< T > >& initial) : tensor_t(initial) {}

        /// \brief Matrix product.
        template <size_t OtherCols>
        typename tensor_t::template convolution_t<1, 0, Cols, OtherCols>
        operator*(const matrix<T, Cols, OtherCols>& other) {
            return sum<1, 0>(*this, other);
        }

        /// \brief Extract a row vector.
        template <size_t I>
        vector<T, Cols> row() { return row(I); }
        vector<T, Cols> row(size_t i) {
            vector<T, Cols> result;
            for (size_t j = 0; j < Cols; ++j)
                result[j] = tensor_t::data[i][j];
            return result;
        }

        /// \brief Extract a column vector.
        template <size_t J>
        vector<T, Rows> col() { return col(J); }
        vector<T, Rows> col(size_t j) {
            vector<T, Rows> result;
            for (size_t i = 0; i < Cols; ++i)
                result[i] = tensor_t::data[i][j];
            return result;
        }
    };
}