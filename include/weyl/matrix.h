#pragma once
#include "weyl/tensor.h"
#include "weyl/vector.h"

namespace weyl
{
    template <typename, size_t, size_t>
    class matrix;

    namespace detail
    {
        template <typename T, size_t Size, size_t i, size_t j>
        struct MinorsElement
        {
            static inline void compute(const matrix<T, Size, Size>& src, matrix<T, Size, Size>& dst) {
                //
                // NOTE: Why does this need to be transposed?
                //       Something is amiss...
                //
                dst[j][i] = src.minor<i, j>().det();
            }
        };

        template <typename T, size_t Size, size_t i, size_t j>
        struct CofactorElement
        {
            static inline void compute(const matrix<T, Size, Size>& src, matrix<T, Size, Size>& dst) {
                const T sign = static_cast<T>(((long)i + (long)j) % 2 ? -1 : 1);
                const T val = sign * src.minor<i, j>().det();
                dst[i][j] = val;
            }
        };

        template <typename T, template<typename, size_t, size_t, size_t> typename ElementT, size_t Size, size_t i, size_t j>
        struct MatrixMap
        {
            static inline void compute(const matrix<T, Size, Size>& src, matrix<T, Size, Size>& dst) {
                horizontal(src, dst);
            }

            static inline void horizontal(const matrix<T, Size, Size>& src, matrix<T, Size, Size>& dst) {
                MatrixMap<T, ElementT, Size, i, j>::vertical(src, dst);
                MatrixMap<T, ElementT, Size, i-1, j>::horizontal(src, dst);
            }

            static inline void vertical(const matrix<T, Size, Size>& src, matrix<T, Size, Size>& dst) {
                MatrixMap<T, ElementT, Size, i, j-1>::vertical(src, dst);
                typename ElementT<T, Size, i, j>::compute(src, dst);
            }
        };

        template <typename T, template<typename, size_t, size_t, size_t> typename ElementT, size_t Size, size_t i>
        struct MatrixMap<T, ElementT, Size, i, 0>
        {
            static inline void vertical(const matrix<T, Size, Size>& src, matrix<T, Size, Size>& dst) {
                typename ElementT<T, Size, i, 0>::compute(src, dst);
            }
        };

        template <typename T, template<typename, size_t, size_t, size_t> typename ElementT, size_t Size, size_t j>
        struct MatrixMap<T, ElementT, Size, 0, j>
        {
            static inline void horizontal(const matrix<T, Size, Size>& src, matrix<T, Size, Size>& dst) {
                MatrixMap<T, ElementT, Size, 0, j>::vertical(src, dst);
            }

            static inline void vertical(const matrix<T, Size, Size>& src, matrix<T, Size, Size>& dst) {
                MatrixMap<T, ElementT, Size, 0, j-1>::vertical(src, dst);
                typename ElementT<T, Size, 0, j>::compute(src, dst);
            }
        };

        template <typename T, template<typename, size_t, size_t, size_t> typename ElementT, size_t Size>
        struct MatrixMap<T, ElementT, Size, 0, 0>
        {
            static inline void vertical(const matrix<T, Size, Size>& src, matrix<T, Size, Size>& dst) {
                typename ElementT<T, Size, 0, 0>::compute(src, dst);
            }
        };
    }

    template <typename T, size_t Rows, size_t Cols>
    class matrix : public tensor<T, Rows, Cols>
    {
    public:

        using tensor_t = tensor<T, Rows, Cols>;

        matrix() : tensor_t() {}

        matrix(const T& value) : tensor_t(static_cast<T>(0)) {
            for (size_t i = 0; i < Rows; ++i)
            for (size_t j = 0; j < Rows; ++j)
                _data[i][j] = value;
        }

        matrix(const tensor<T, Rows, Cols>& tens) : tensor_t(tens) {}

        matrix(const std::initializer_list< std::initializer_list< T > >& initial) : tensor_t(initial) {}

        /// \brief Matrix product.
        template <size_t OtherCols>
        typename tensor_t::template convolution_t<1, 0, Cols, OtherCols>
        operator*(const tensor<T, Cols, OtherCols>& other) const {
            return sum<1, 0>(*this, other);
        }

        /// \brief Matrix*vector product.
        vector<T, Rows>
        operator*(const vector<T, Cols>& other) const {
            return sum<1, 0>(*this, other);
        }

        /// \brief Directly set a row
        template <size_t I>
        void set_row(const vector<T, Cols>& row) { set_row(I, row); }
        void set_row(size_t i, const vector<T, Cols>& row) {
            for (size_t j = 0; j < Cols; ++j)
                _data[i][j] = row[j];
        }

        /// \brief Directly set a column
        template <size_t J>
        void set_col(const vector<T, Rows>& col) { set_col(J, col); }
        void set_col(size_t j, const vector<T, Rows>& col) {
            for (size_t i = 0; i < Cols; ++i)
                _data[i][j] = col[i];
        }

        /// \brief Extract a row vector.
        template <size_t I>
        vector<T, Cols> row() const { return row(I); }
        vector<T, Cols> row(size_t i) const {
            vector<T, Cols> result;
            for (size_t j = 0; j < Cols; ++j)
                result[j] = _data[i][j];
            return result;
        }

        /// \brief Extract a column vector.
        template <size_t J>
        vector<T, Rows> col() const { return col(J); }
        vector<T, Rows> col(size_t j) const {
            vector<T, Rows> result;
            for (size_t i = 0; i < Cols; ++i)
                result[i] = _data[i][j];
            return result;
        }

        /// \brief Return the transposed matrix.
        matrix<T, Cols, Rows> transpose() const {
            matrix<T, Cols, Rows> result;
            for (size_t i = 0; i < Rows; ++i)
            for (size_t j = 0; j < Cols; ++j)
                result[j][i] = _data[i][j];
            return result;
        }

        /// \brief Return the minor matrix (with col I and row J removed)
        template<size_t I, size_t J>
        matrix<T, Rows-1, Cols-1> minor() const {
            matrix<T, Rows-1, Cols-1> m;
            for (size_t i = 0; i < I; ++i)
            for (size_t j = 0; j < J; ++j)
                m[i][j] = _data[i][j];
            for (size_t i = I+1; i < Rows; ++i)
            for (size_t j = J+1; j < Cols; ++j)
                m[i-1][j-1] = _data[i][j];
            return m;
        }
    };


    template <typename T, size_t Size>
    class matrix<T, Size, Size> : public tensor<T, Size, Size>
    {
    public:

        using tensor_t = tensor<T, Size, Size>;

        matrix(const T& value = static_cast<T>(1)) : tensor_t(static_cast<T>(0)) {
            for (size_t i = 0; i < Size; ++i)
                _data[i][i] = value;
        }

        matrix(const tensor<T, Size, Size>& tens) : tensor_t(tens) {}

        matrix(const std::initializer_list< std::initializer_list< T > >& initial) : tensor_t(initial) {}

        /// \brief Matrix product.
        template <size_t OtherCols>
        typename tensor_t::template convolution_t<1, 0, Size, OtherCols>
        operator*(const tensor<T, Size, OtherCols>& other) const {
            return sum<1, 0>(*this, other);
        }

        /// \brief Matrix*vector product.
        vector<T, Size>
        operator*(const vector<T, Size>& other) const {
            return sum<1, 0>(*this, other);
        }

        matrix<T, Size, Size>
        operator*(const T& value) const {
            return tensor_t::operator*(value);
        }

        /// \brief Directly set a row
        template <size_t I>
        void set_row(const vector<T, Size>& row) { set_row(I, row); }
        void set_row(size_t i, const vector<T, Size>& row) {
            for (size_t j = 0; j < Size; ++j)
                _data[i][j] = row[j];
        }

        /// \brief Directly set a column
        template <size_t J>
        void set_col(const vector<T, Size>& col) { set_col(J, col); }
        void set_col(size_t j, const vector<T, Size>& col) {
            for (size_t i = 0; i < Size; ++i)
                _data[i][j] = col[i];
        }

        /// \brief Extract a row vector.
        template <size_t I>
        vector<T, Size> row() const { return row(I); }
        vector<T, Size> row(size_t i) const {
            vector<T, Size> result;
            for (size_t j = 0; j < Size; ++j)
                result[j] = _data[i][j];
            return result;
        }

        /// \brief Extract a column vector.
        template <size_t J>
        vector<T, Size> col() const { return col(J); }
        vector<T, Size> col(size_t j) const {
            vector<T, Size> result;
            for (size_t i = 0; i < Size; ++i)
                result[i] = _data[i][j];
            return result;
        }

        /// \brief Return the transposed matrix.
        matrix<T, Size, Size> transpose() const {
            matrix<T, Size, Size> result;
            for (size_t i = 0; i < Size; ++i)
            for (size_t j = 0; j < Size; ++j)
                result[j][i] = _data[i][j];
            return result;
        }

        /// \brief Return the minor matrix (with col I and row J removed)
        template<size_t I, size_t J>
        matrix<T, Size-1, Size-1> minor() const {
            matrix<T, Size-1, Size-1> m;
            for (size_t i = 0; i < Size; ++i)
            for (size_t j = 0; j < Size; ++j)
                if (i != I && j != J)
                    m[i < I ? i : i - 1][j < J ? j : j - 1] = _data[i][j];
            return m;
        }

        /// \brief Get the determinant of the matrix
        T det() const {
            T result = static_cast<T>(0);
            //
            // TODO: We only need one row or column of the cofactor matrix!
            //       Significant computational reduction is possible.
            //
            const matrix<T, Size, Size> C = cofactor();
            for (size_t i = 0; i < Size; ++i)
                result += _data[i][0] * C[i][0];
            return result;
        }

        matrix<T, Size, Size> minors() const {
            matrix<T, Size, Size> result;
            detail::MatrixMap<T, detail::MinorsElement, Size, Size - 1, Size - 1>::compute(*this, result);
            return result;
        }

        matrix<T, Size, Size> cofactor() const {
            matrix<T, Size, Size> result;
            detail::MatrixMap<T, detail::CofactorElement, Size, Size - 1, Size - 1>::compute(*this, result);
            return result;
        }

        matrix<T, Size, Size> adj() const {
            return cofactor().transpose();
        }

        /// \brief Return the inverse of the matrix.
        matrix<T, Size, Size> inverse() const {
            matrix<T, Size, Size> result(static_cast<T>(1));
            return adj() * (static_cast<T>(1) / det());
        }
    };


    template <typename T>
    class matrix<T, 2, 2> : public tensor<T, 2, 2>
    {
    public:

        using tensor_t = tensor<T, 2, 2>;

        matrix() : tensor_t() {}

        matrix(const T& value) : tensor_t(static_cast<T>(0)) {
            for (size_t i = 0; i < 2; ++i)
                _data[i][i] = value;
        }

        matrix(const tensor<T, 2, 2>& tens) : tensor_t(tens) {}

        matrix(const std::initializer_list< std::initializer_list< T > >& initial) : tensor_t(initial) {}

        /// \brief Matrix product.
        template <size_t OtherCols>
        typename tensor_t::template convolution_t<1, 0, 2, OtherCols>
        operator*(const tensor<T, 2, OtherCols>& other) const {
            return sum<1, 0>(*this, other);
        }

        /// \brief Matrix*vector product.
        vector<T, 2>
        operator*(const vector<T, 2>& other) const {
            return sum<1, 0>(*this, other);
        }

        /// \brief Directly set a row
        template <size_t I>
        void set_row(const vector<T, 2>& row) { set_row(I, row); }
        void set_row(size_t i, const vector<T, 2>& row) {
            for (size_t j = 0; j < 2; ++j)
                _data[i][j] = row[j];
        }

        /// \brief Directly set a column
        template <size_t J>
        void set_col(const vector<T, 2>& col) { set_col(J, col); }
        void set_col(size_t j, const vector<T, 2>& col) {
            for (size_t i = 0; i < 2; ++i)
                _data[i][j] = col[i];
        }

        /// \brief Extract a row vector.
        template <size_t I>
        vector<T, 2> row() const { return row(I); }
        vector<T, 2> row(size_t i) const {
            vector<T, 2> result;
            for (size_t j = 0; j < 2; ++j)
                result[j] = _data[i][j];
            return result;
        }

        /// \brief Extract a column vector.
        template <size_t J>
        vector<T, 2> col() const { return col(J); }
        vector<T, 2> col(size_t j) const {
            vector<T, 2> result;
            for (size_t i = 0; i < 2; ++i)
                result[i] = _data[i][j];
            return result;
        }

        /// \brief Return the transposed matrix.
        matrix<T, 2, 2> transpose() const {
            matrix<T, 2, 2> result;
            for (size_t i = 0; i < 2; ++i)
            for (size_t j = 0; j < 2; ++j)
                result[j][i] = _data[i][j];
            return result;
        }

        /// \brief Return the minor matrix (with col I and row J removed)
        template<size_t I, size_t J>
        matrix<T, 1, 1> minor() const {
            matrix<T, 1, 1> m;
            for (size_t i = 0; i < I; ++i)
            for (size_t j = 0; j < J; ++j)
                m[i][j] = _data[i][j];
            for (size_t i = I+1; i < 2; ++i)
            for (size_t j = J+1; j < 2; ++j)
                m[i-1][j-1] = _data[i][j];
            return m;
        }

        /// \brief Get the determinant of the matrix
        T det() const {
            return (_data[0][0] * _data[1][1]) - (_data[1][0] * _data[0][1]);
        }

        matrix<T, 2, 2> minors() const {
            return matrix<T, 2, 2>({
                { _data[1][1], _data[1][0] },
                { _data[0][1], _data[0][0] }
            });
        }

        matrix<T, 2, 2> cofactor() const {
            return matrix<T, 2, 2>({
                { _data[1][1], -_data[1][0] },
                { -_data[0][1], _data[0][0] }
            });
        }

        matrix<T, 2, 2> adj() const {
            return matrix<T, 2, 2>({
                { _data[1][1], -_data[0][1] },
                { -_data[1][0], _data[0][0] }
            });
        }

        /// \brief Return the inverse of the matrix.
        matrix<T, 2, 2> inverse() const {
            T det_inv = static_cast<T>(1.0f) / det();
            return matrix<T, 2, 2>({
                { det_inv * _data[1][1], -det_inv * _data[0][1] },
                { -det_inv * _data[1][0], det_inv * _data[0][0] }
            });
        }
    };
}
