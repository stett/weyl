#pragma once
#ifndef _WEYL_TENSOR_H_
#define _WEYL_TENSOR_H_

#include <initializer_list>
#include <algorithm>
#include <type_traits>

namespace weyl
{
    template <typename T, size_t N0, size_t... N>
    class tensor;

    /// \class vector<T, N>
    /// \brief Rank-one tensors are aliased as vectors.
    template <typename T, size_t N>
    using vector = tensor<T, N>;

    /// \class matrix<T, Rows, Cols>
    /// \brief Rank-two tensors are aliased as matrices.
    template <typename T, size_t Rows, size_t Cols>
    using matrix = tensor<T, Rows, Cols>;

    namespace detail
    {
        /// \struct Dimension
        /// \brief Given a set of dimensions N_0 through N_m, pick the Ith one and house its depth in value.
        ///
        /// In other words, Dimension<1, 8, 13, 22>::value == 8
        template <size_t I, size_t N0, size_t... N>
        struct Dimension
        {
            enum { value = Dimension<I-1, N...>::value };
        };

        template <size_t N0, size_t... N>
        struct Dimension<0, N0, N...>
        {
            enum { value = N0 };
        };

        /// \struct Rank
        /// \brief Given a set of dimensions N_0 through N_m, house the rank, m.
        template <size_t N0, size_t... N>
        struct Rank
        {
            enum { value = 1 + (int)Rank<N...>::value };
        };

        template <size_t N>
        struct Rank<N>
        {
            enum { value = 1 };
        };

        /// \struct Reduction
        /// \brief Given a set of dimensions N through M, remove the Ith one, and house a tensor of the resulting dimensionality.
        template <size_t I, template<size_t...> class T, size_t... N>
        struct Reduction
        {
            template <size_t NM, size_t... M>
            using reduced_t = typename Reduction<I-1, T, N..., NM>::template reduced_t<M...>;
        };

        template <template<size_t...> class T, size_t... N>
        struct Reduction<0, T, N...>
        {
            template <size_t NM, size_t... M>
            using reduced_t = T<N..., M...>;
        };

        /// \struct ReducibleTensor
        /// \brief Provides a wrapper around the tensor template which may be used as a parameter for Reduction
        template <size_t... N>
        struct ReducibleTensor
        {
            template <typename T>
            using tensor_t = tensor<T, N...>;
        };

        template <>
        struct ReducibleTensor<>
        {
            template <typename T>
            using tensor_t = T;
        };

        /// \struct DoubleReducibleTensor
        /// \brief Provides a second rank reduction, for when two indexes must be removed from a tensor type.
        template <size_t J>
        struct DoubleReducibleTensor
        {
            template <size_t... N>
            struct Sub
            {
                // Note that in "J-1", the offset by one corrects for the fact that we are storing
                // the index of the second reduction in the parameter pack which is reduced by the
                // first one. Thus it will be one off.
                template <typename T>
                using tensor_t = typename Reduction< J - 1, ReducibleTensor >::template reduced_t<N...>::template tensor_t<T>;
            };
        };

        /// \struct TensorConvolution
        /// \brief Uses double rank reduction to determine the resultant type of a tensor convolution.
        template <size_t Rank, size_t I, size_t J, typename T, size_t... N>
        struct TensorConvolution
        {
            using tensor_t = typename Reduction< I, DoubleReducibleTensor< Rank + J >::template Sub >::template reduced_t<N...>::template tensor_t<T>;
        };

        /// \struct Sum
        /// \brief Perform one iteration of a tensor convolution. This template is not intended
        /// for use outside of Weyl, but is exposed for those who want to use it for special
        /// optimizations or use cases.
        /// 
        /// The static member `partial` must be invoked once per dimension for chosen index pair I, J,
        /// with the same `result` tensor. On the first iteration, the result tensor should be
        /// populated with zeros.
        template <size_t OuterI, size_t OuterJ, size_t I, size_t J, size_t RankN, size_t RankM>
        struct Sum
        {
            /// \brief Break down a full tensor summation into individual scalar product "parts".
            /// \param result The resulting tensor.
            /// \param part_a The decomposed sub-tensor of the first operand tensor.
            /// \param part_b The decomposed sub-tensor of the second operand tensor.
            /// \param a The first operand tensor.
            /// \param b The second operand tensor.
            /// \param inner The current value of the index pair.
            template <typename ResultT, typename PartA, typename PartB, typename T, size_t... N, size_t... M>
            inline static void partial(ResultT& result, const PartA& part_a, const PartB& part_b, const tensor<T, N...>& a, const tensor<T, M...>& b, size_t inner) {
                for (size_t i = 0; i < detail::Dimension<OuterI, N...>::value; ++i)
                    Sum< OuterI+1, OuterJ, I, J, RankN, RankM >::partial(result[i], part_a[i], part_b, a, b, inner);
            }
        };

        template <size_t OuterJ, size_t I, size_t J, size_t RankN, size_t RankM>
        struct Sum< I, OuterJ, I, J, RankN, RankM >
        {
            template <typename ResultT, typename PartA, typename PartB, typename T, size_t... N, size_t... M>
            inline static void partial(ResultT& result, const PartA& part_a, const PartB& part_b, const tensor<T, N...>& a, const tensor<T, M...>& b, size_t inner) {
                Sum< I+1, OuterJ, I, J, RankN, RankM >::partial(result, part_a[inner], part_b, a, b, inner);
            }
        };

        template <size_t OuterJ, size_t I, size_t J, size_t RankN, size_t RankM>
        struct Sum< RankN, OuterJ, I, J, RankN, RankM >
        {
            template <typename ResultT, typename PartA, typename PartB, typename T, size_t... N, size_t... M>
            inline static void partial(ResultT& result, const PartA& part_a, const PartB& part_b, const tensor<T, N...>& a, const tensor<T, M...>& b, size_t inner) {
                for (size_t j = 0; j < detail::Dimension<OuterJ, M...>::value; ++j)
                    Sum< RankN, OuterJ+1, I, J, RankN, RankM >::partial(result[j], part_a, part_b[j], a, b, inner);
            }
        };

        template <size_t I, size_t J, size_t RankN, size_t RankM>
        struct Sum< RankN, J, I, J, RankN, RankM >
        {
            template <typename ResultT, typename PartA, typename PartB, typename T, size_t... N, size_t... M>
            inline static void partial(ResultT& result, const PartA& part_a, const PartB& part_b, const tensor<T, N...>& a, const tensor<T, M...>& b, size_t inner) {
                Sum< RankN, J+1, I, J, RankN, RankM >::partial(result, part_a, part_b[inner], a, b, inner);
            }
        };

        template <size_t I, size_t J, size_t RankN, size_t RankM>
        struct Sum< RankN, RankM, I, J, RankN, RankM >
        {
            template <typename ResultT, typename PartA, typename PartB, typename T, size_t... N, size_t... M>
            inline static void partial(ResultT& result, const PartA& part_a, const PartB& part_b, const tensor<T, N...>& a, const tensor<T, M...>& b, size_t inner) {
                result += part_a * part_b;
            }
        };

        template <typename T, size_t Size, size_t i, size_t j>
        struct MinorsElement
        {
            static inline void compute(const matrix<T, Size, Size>& src, matrix<T, Size, Size>& dst) {
                //
                // NOTE: Why does this need to be transposed?
                //       Something is amiss...
                //
                dst[j][i] = src.template minor<i, j>().det();
            }
        };

        template <typename T, size_t Size, size_t i, size_t j>
        struct CofactorElement
        {
            static inline void compute(const matrix<T, Size, Size>& src, matrix<T, Size, Size>& dst) {
                const T sign = static_cast<T>(((long)i + (long)j) % 2 ? -1 : 1);
                const T val = sign * src.template minor<i, j>().det();
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
                ElementT<T, Size, i, j>::compute(src, dst);
            }
        };

        template <typename T, template<typename, size_t, size_t, size_t> typename ElementT, size_t Size, size_t i>
        struct MatrixMap<T, ElementT, Size, i, 0>
        {
            static inline void vertical(const matrix<T, Size, Size>& src, matrix<T, Size, Size>& dst) {
                ElementT<T, Size, i, 0>::compute(src, dst);
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
                ElementT<T, Size, 0, j>::compute(src, dst);
            }
        };

        template <typename T, template<typename, size_t, size_t, size_t> typename ElementT, size_t Size>
        struct MatrixMap<T, ElementT, Size, 0, 0>
        {
            static inline void vertical(const matrix<T, Size, Size>& src, matrix<T, Size, Size>& dst) {
                ElementT<T, Size, 0, 0>::compute(src, dst);
            }
        };
    }

    /// \brief Produce the tensor which is the convolution of two tensors.
    template <size_t I, size_t J, typename T, size_t N0, size_t... N, size_t M0, size_t... M>
    typename tensor<T, N0, N...>::template convolution_t<I, J, M0, M...>
    sum(const tensor<T, N0, N...>& a, const tensor<T, M0, M...>& b) {

        // Determine the resulting tensor type, create an instance of it,
        // and initialize all of its values to "zero".
        using tensor_t = typename tensor<T, N0, N...>::template convolution_t<I, J, M0, M...>;
        tensor_t result(static_cast<T>(0));

        // Do the summation - this loop corresponds to the Riemann sum in an ordinary tensor product.
        for (size_t inner = 0; inner < detail::Dimension<I, N0, N...>::value; ++inner)
            detail::Sum< 0, 0, I, J, detail::Rank<N0, N...>::value, detail::Rank<M0, M...>::value >::partial(result, a, b, a, b, inner);

        // Return the convolution
        return result;
    }

    /// \brief Degenerate convolution case - mirrors a simple vector dot product.
    template <size_t I, size_t J, typename T, size_t N, size_t M>
    T sum(const tensor<T, N>& a, const tensor<T, M>& b) {
        static_assert(I == 0 && J == 0, "Both tensors have rank 1, so the first dimensions of each must be used in the convolution.");
        T result = static_cast<T>(0);
        for (size_t inner = 0; inner < N; ++inner)
            detail::Sum< 0, 0, I, J, 1, 1 >::partial(result, a, b, a, b, inner);
        return result;
    }

    /// \brief Scalar tensor product.
    template <typename T, size_t... N>
    tensor<T, N...> operator*(const T& value, const tensor<T, N...>& tens) {
        return tens * value;
    }

    /* TODO: Populate this function...
    /// \brief Scalar "division" by tensor... do it element-wise
    template <typename T, size_t... N>
    tensor<T, N...> operator/(const T& value, const tensor<T, N...>& tens) {
        return 
    }
    */

    /// \brief Inner product for rank-N tensors along the D'th dimension.
    ///
    /// This amounts to a dot product for rank-1 tensors.
    template <typename T, size_t... N, size_t D=0>
    typename tensor<T, N...>::template convolution_t<D, D, N...>
    inner(const tensor<T, N...>& a, const tensor<T, N...>& b) {
        return sum<D, D>(a, b);
    }

    /// \brief concatenation of two tensors of the same rank.
    ///
    /// The result is a tensor whos rank is equal to that of both of the
    /// operands, and whose dimensionality is the sum of them. Elements of
    /// the second are concatenated "diagonally" to the first.
    //
    // TODO: Templatize this to generalize for N concatenations!
    //
    template <size_t N0, size_t N1>
    tensor<float_t, N0 + N1>
    cat(const tensor<float_t, N0>& a, const tensor<float_t, N1>& b) {
        tensor<float_t, N0 + N1> result;
        for (size_t i = 0; i < N0; ++i)
            result[i] = a[i];
        for (size_t i = 0; i < N1; ++i)
            result[i + N0] = b[i];
        return result;
    }

    template <size_t N0, size_t N1, size_t N2>
    tensor<float_t, N0 + N1 + N2>
    cat(const tensor<float_t, N0>& a, const tensor<float_t, N1>& b, const tensor<float_t, N2>& c) {
        return cat(cat(a, b), c);
    }

    template <size_t N0, size_t N1, size_t N2, size_t N3>
    tensor<float_t, N0 + N1 + N2 + N3>
    cat(const tensor<float_t, N0>& a, const tensor<float_t, N1>& b, const tensor<float_t, N2>& c, const tensor<float_t, N3>& d) {
        return cat(cat(a, b, c), d);
    }

    template <typename T, size_t N0, size_t M0, size_t N1, size_t M1>
    tensor<T, N0 + M0, N1 + M1>
    cat(const tensor<T, N0, M0>& a, const tensor<T, N1, M1>& b, const T& default_value = static_cast<T>(0)) {
        tensor<T, N0 + N1, M0 + M1> result(default_value);
        for (size_t i = 0; i < N0; ++i)
        for (size_t j = 0; j < M0; ++j)
            result[i][j] = a[i][j];
        for (size_t i = 0; i < N1; ++i)
        for (size_t j = 0; j < M1; ++j)
            result[N0 + i][M0 + j] = b[i][j];
        return result;
    }

    template <typename T, size_t N>
    T dot(const vector<T, N>& a, const vector<T, N>& b) {
        return sum<0, 0>(a, b);
    }

    template <typename T>
    T cross(const vector<T, 2>& a, const vector<T, 2>& b) {
        return (a[0] * b[1]) - (a[1] * b[0]);
    }

    template <typename T>
    vector<T, 3> cross(const vector<T, 3>& a, const vector<T, 3>& b) {
        return vector<T, 3>({
            (a[1] * b[2]) - (a[2] * b[1]),
            (a[2] * b[0]) - (a[0] * b[2]),
            (a[0] * b[1]) - (a[1] * b[0])
        });
    }

    /// \class tensor
    /// \brief A tensor of arbitrary rank and dimension, the core Weyl class.
    ///
    /// All classes in Weyl are specializations of tensor.
    template <typename T, size_t N0, size_t... N>
    class tensor
    {
    public:

        /// \brief Get the number of indexes of the tensor
        using rank = detail::Rank<N0, N...>;

        /// \brief Get the number of dimensions represented by the Ith index of the tensor.
        template <size_t I>
        using dimension = detail::Dimension<I, N0, N...>;

        using initializer_t = typename std::initializer_list< typename tensor<T, N...>::initializer_t >;

        using tensor_t = tensor<T, N0, N...>;

        using data_t = tensor<T, N...>[N0];

        using element_t = T;

        /// \brief Determine the tensor type which would result from convolution with another tensor.
        template <size_t I, size_t J, size_t... M>
        using convolution_t = typename detail::TensorConvolution< detail::Rank<N0, N...>::value, I, J, T, N0, N..., M... >::tensor_t;

        tensor() {}

        tensor(const T& value) {
            std::fill(_data, _data + N0, tensor<T, N...>(value));
        }

        tensor(const initializer_t& values) {
            std::copy(values.begin(), values.end(), _data);
        }

        tensor(const tensor_t& other) {
            std::copy(other._data, other._data + N0, _data);
        }

        tensor<T, N...>& operator[](size_t i) {
            return _data[i];
        }

        const tensor<T, N...>& operator[](size_t i) const {
            return _data[i];
        }

        bool operator==(const tensor_t& other) const {
            bool result = true;
            for (size_t i = 0; i < N0; ++i)
                if (_data[i] != other._data[i])
                    return false;
            return true;
        }

        bool operator!=(const tensor_t& other) const {
            return !operator==(other);
        }

        tensor_t& operator*=(const T& value) {
            for (size_t i = 0; i < N0; ++i)
                _data[i] *= value;
            return *this;
        }

        tensor_t operator*(const T& value) const {
            tensor_t result(*this);
            result *= value;
            return result;
        }

        tensor_t& operator/=(const T& value) {
            for (size_t i = 0; i < N0; ++i)
                _data[i] /= value;
            return *this;
        }

        tensor_t operator/(const T& value) const {
            tensor_t result(*this);
            result /= value;
            return result;
        }

        tensor_t& operator+=(const tensor_t& other) {
            for (size_t i = 0; i < N0; ++i)
                _data[i] += other._data[i];
            return *this;
        }

        tensor_t operator+(const tensor_t& other) const {
            tensor_t result(*this);
            result += other;
            return result;
        }

        tensor_t& operator-=(const tensor_t& other) {
            for (size_t i = 0; i < N0; ++i)
                _data[i] -= other._data[i];
            return *this;
        }

        tensor_t operator-(const tensor_t& other) const {
            tensor_t result(*this);
            result -= other;
            return result;
        }

        tensor_t operator-() const {
            tensor_t result;
            for (size_t i = 0; i < N0; ++i)
                result._data[i] = -_data[i];
            return result;
        }

        tensor<T, N...>* data() {
            return _data;
        }

        const tensor<T, N...>* data() const {
            return _data;
        }

    protected:
        data_t _data;
    };

    /// \class tensor<T, Rows, Cols>
    /// \brief Rank-two template for the tensor class.
    /// 
    /// Matrix-style methods have been added to this specialization
    template <typename T, size_t Rows, size_t Cols>
    class tensor<T, Rows, Cols>
    {
    public:

        /// \brief Get the number of indexes of the tensor. In this case ::value will always be 2.
        using rank = detail::Rank<Rows, Cols>;

        /// \brief The number of dimensions represented by the Ith index.
        template <size_t I>
        using dimension = detail::Dimension<I, Rows, Cols>;

        // NOTE: Should this take rows??? wouldn't that be nicer?
        using initializer_t = typename std::initializer_list< typename tensor<T, Cols>::initializer_t >;

        using tensor_t = tensor<T, Rows, Cols>;

        using data_t = tensor<T, Cols>[Rows];

        using element_t = T;

        /// \brief Determine the tensor type which would result from convolution with another tensor.
        template <size_t I, size_t J, size_t... M>
        using convolution_t = typename detail::TensorConvolution< detail::Rank<Rows, Cols>::value, I, J, T, Rows, Cols, M... >::tensor_t;

        tensor() {}

        tensor(const T& value) {
            std::fill(_data, _data + Rows, tensor<T, Cols>(value));
        }

        tensor(const initializer_t& values) {
            std::copy(values.begin(), values.end(), _data);
        }

        tensor(const tensor_t& other) {
            std::copy(other._data, other._data + Rows, _data);
        }

        /// \brief Initialize from a vector on the diagonal
        tensor(const tensor<T, Rows>& diag) {
            static_assert(Rows == Cols, "Must be square matrix to initialize with a diagonal vector");
            std::fill(_data, _data + Rows, tensor<T, Cols>(static_cast<T>(0)));
            set_diag(diag);
        }

        tensor<T, Cols>& operator[](size_t i) {
            return _data[i];
        }

        const tensor<T, Cols>& operator[](size_t i) const {
            return _data[i];
        }

        bool operator==(const tensor_t& other) const {
            bool result = true;
            for (size_t i = 0; i < Rows; ++i)
                if (_data[i] != other._data[i])
                    return false;
            return true;
        }

        bool operator!=(const tensor_t& other) const {
            return !operator==(other);
        }

        tensor_t& operator*=(const T& value) {
            for (size_t i = 0; i < Rows; ++i)
                _data[i] *= value;
            return *this;
        }

        tensor_t operator*(const T& value) const {
            tensor_t result(*this);
            result *= value;
            return result;
        }

        tensor_t& operator/=(const T& value) {
            for (size_t i = 0; i < Rows; ++i)
                _data[i] /= value;
            return *this;
        }

        tensor_t operator/(const T& value) const {
            tensor_t result(*this);
            result /= value;
            return result;
        }

        tensor_t& operator+=(const tensor_t& other) {
            for (size_t i = 0; i < Rows; ++i)
                _data[i] += other._data[i];
            return *this;
        }

        tensor_t operator+(const tensor_t& other) const {
            tensor_t result(*this);
            result += other;
            return result;
        }

        tensor_t& operator-=(const tensor_t& other) {
            for (size_t i = 0; i < Rows; ++i)
                _data[i] -= other._data[i];
            return *this;
        }

        tensor_t operator-(const tensor_t& other) const {
            tensor_t result(*this);
            result -= other;
            return result;
        }

        tensor_t operator-() const {
            tensor_t result;
            for (size_t i = 0; i < Rows; ++i)
                result._data[i] = -_data[i];
            return result;
        }

        tensor<T, Cols>* data() {
            return _data;
        }

        const tensor<T, Cols>* data() const {
            return _data;
        }

        // Matrix operations

        /// \brief Matrix product.
        template <size_t OtherCols>
        typename tensor_t::template convolution_t<1, 0, Cols, OtherCols>
        operator*(const tensor<T, Cols, OtherCols>& other) const {
            return sum<1, 0>(*this, other);
        }

        /// \brief Matrix*vector product.
        tensor<T, Rows>
        operator*(const tensor<T, Cols>& other) const {
            return sum<1, 0>(*this, other);
        }

        /// \brief Directly set a row
        template <size_t I>
        void set_row(const tensor<T, Cols>& row) { set_row(I, row); }
        void set_row(size_t i, const tensor<T, Cols>& row) {
            for (size_t j = 0; j < Cols; ++j)
                _data[i][j] = row[j];
        }

        /// \brief Directly set a column
        template <size_t J>
        void set_col(const tensor<T, Rows>& col) { set_col(J, col); }
        void set_col(size_t j, const tensor<T, Rows>& col) {
            for (size_t i = 0; i < Cols; ++i)
                _data[i][j] = col[i];
        }

        /// \brief Extract a row vector.
        template <size_t I>
        tensor<T, Cols> row() const { return row(I); }
        tensor<T, Cols> row(size_t i) const {
            tensor<T, Cols> result;
            for (size_t j = 0; j < Cols; ++j)
                result[j] = _data[i][j];
            return result;
        }

        /// \brief Extract a column vector.
        template <size_t J>
        tensor<T, Rows> col() const { return col(J); }
        tensor<T, Rows> col(size_t j) const {
            tensor<T, Rows> result;
            for (size_t i = 0; i < Cols; ++i)
                result[i] = _data[i][j];
            return result;
        }

        /// \brief Return the transposed matrix.
        tensor<T, Cols, Rows> transpose() const {
            tensor<T, Cols, Rows> result;
            for (size_t i = 0; i < Rows; ++i)
            for (size_t j = 0; j < Cols; ++j)
                result[j][i] = _data[i][j];
            return result;
        }

        /// \brief Return the minor matrix (with col I and row J removed)
        template<size_t I, size_t J>
        tensor<T, Rows-1, Cols-1> minor() const {
            tensor<T, Rows-1, Cols-1> m;
            for (size_t i = 0; i < I; ++i)
            {
                for (size_t j = 0; j < J; ++j)
                    m[i][j] = _data[i][j];
                for (size_t j = J+1; j < Cols; ++j)
                    m[i][j-1] = _data[i][j];
            }
            for (size_t i = I+1; i < Rows; ++i)
            {
                for (size_t j = 0; j < J; ++j)
                    m[i-1][j] = _data[i][j];
                for (size_t j = J+1; j < Cols; ++j)
                    m[i-1][j-1] = _data[i][j];
            }
            return m;
        }

        /// \brief Return a vector containing concatenated rows
        tensor<T, Rows * Cols> rows() const {
            tensor<T, Rows * Cols> result;
            for (size_t i = 0; i < Rows; ++i)
            for (size_t j = 0; j < Cols; ++j)
                result[i + (i * j)] = _data[i][j];
            return result;
        }

        /// \brief Return a vector containing concatenated cols
        tensor<T, Rows * Cols> cols() const {
            tensor<T, Rows * Cols> result;
            for (size_t j = 0; j < Cols; ++j)
            for (size_t i = 0; i < Rows; ++i)
                result[i + (i * j)] = _data[i][j];
            return result;
        }

        // Square matrix operations

        template<size_t Size=Rows>
        static constexpr typename std::enable_if<Size == Rows && Size == Cols, tensor_t>::type identity()
        {
            return tensor_t(tensor<T, Size>(static_cast<T>(1)));
        }

        /// \brief Directly set the diagonal
        template <size_t Size=Rows>
        typename std::enable_if<Size == Rows && Size == Cols, void>::type
        set_diag(const tensor<T, Size>& diag) {
            for (size_t i = 0; i < Size; ++i)
                _data[i][i] = diag[i];
        }

        /// \brief Get the determinant of the matrix
        template <size_t Size=Rows>
        typename std::enable_if<Size == Rows && Size == Cols, T>::type
        det() const {
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

        template <size_t Size=Rows>
        typename std::enable_if<Size == Rows && Size == Cols, matrix<T, Size, Size>>::type
        minors() const {
            matrix<T, Size, Size> result;
            detail::MatrixMap<T, detail::MinorsElement, Size, Size - 1, Size - 1>::compute(*this, result);
            return result;
        }

        template <size_t Size=Rows>
        typename std::enable_if<Size == Rows && Size == Cols, matrix<T, Size, Size>>::type
        cofactor() const {
            matrix<T, Size, Size> result;
            detail::MatrixMap<T, detail::CofactorElement, Size, Size - 1, Size - 1>::compute(*this, result);
            return result;
        }

        template <size_t Size=Rows>
        typename std::enable_if<Size == Rows && Size == Cols, matrix<T, Size, Size>>::type
        adj() const {
            return cofactor().transpose();
        }

        /// \brief Return the inverse of the matrix.
        template <size_t Size=Rows>
        typename std::enable_if<Size == Rows && Size == Cols, matrix<T, Size, Size>>::type
        inverse() const {
            matrix<T, Size, Size> result(static_cast<T>(1));
            return adj() * (static_cast<T>(1) / det());
        }

    protected:
        data_t _data;
    };

    template <typename T>
    class tensor<T, 2, 2>
    {
    public:

        static constexpr size_t Rows = 2;
        static constexpr size_t Cols = 2;

        /// \brief Get the number of indexes of the tensor. In this case ::value will always be 2.
        using rank = detail::Rank<Rows, Cols>;

        /// \brief The number of dimensions represented by the Ith index.
        template <size_t I>
        using dimension = detail::Dimension<I, Rows, Cols>;

        // NOTE: Should this take rows??? wouldn't that be nicer?
        using initializer_t = typename std::initializer_list< typename tensor<T, Cols>::initializer_t >;

        using tensor_t = tensor<T, Rows, Cols>;

        using data_t = tensor<T, Cols>[Rows];

        using element_t = T;

        /// \brief Determine the tensor type which would result from convolution with another tensor.
        template <size_t I, size_t J, size_t... M>
        using convolution_t = typename detail::TensorConvolution< detail::Rank<Rows, Cols>::value, I, J, T, Rows, Cols, M... >::tensor_t;

        tensor() {}

        tensor(const T& value) {
            std::fill(_data, _data + Rows, tensor<T, Cols>(value));
        }

        tensor(const initializer_t& values) {
            std::copy(values.begin(), values.end(), _data);
        }

        tensor(const tensor_t& other) {
            std::copy(other._data, other._data + Rows, _data);
        }

        tensor(const tensor<T, 2>& diag) {
            _data[0][0] = diag[0];
            _data[1][1] = diag[1];
            _data[0][1] = static_cast<T>(0);
            _data[1][0] = static_cast<T>(0);
        }

        tensor<T, Cols>& operator[](size_t i) {
            return _data[i];
        }

        const tensor<T, Cols>& operator[](size_t i) const {
            return _data[i];
        }

        bool operator==(const tensor_t& other) const {
            bool result = true;
            for (size_t i = 0; i < Rows; ++i)
                if (_data[i] != other._data[i])
                    return false;
            return true;
        }

        bool operator!=(const tensor_t& other) const {
            return !operator==(other);
        }

        tensor_t& operator*=(const T& value) {
            for (size_t i = 0; i < Rows; ++i)
                _data[i] *= value;
            return *this;
        }

        tensor_t operator*(const T& value) const {
            tensor_t result(*this);
            result *= value;
            return result;
        }

        tensor_t& operator/=(const T& value) {
            for (size_t i = 0; i < Rows; ++i)
                _data[i] /= value;
            return *this;
        }

        tensor_t operator/(const T& value) const {
            tensor_t result(*this);
            result /= value;
            return result;
        }

        tensor_t& operator+=(const tensor_t& other) {
            for (size_t i = 0; i < Rows; ++i)
                _data[i] += other._data[i];
            return *this;
        }

        tensor_t operator+(const tensor_t& other) const {
            tensor_t result(*this);
            result += other;
            return result;
        }

        tensor_t& operator-=(const tensor_t& other) {
            for (size_t i = 0; i < Rows; ++i)
                _data[i] -= other._data[i];
            return *this;
        }

        tensor_t operator-(const tensor_t& other) const {
            tensor_t result(*this);
            result -= other;
            return result;
        }

        tensor_t operator-() const {
            tensor_t result;
            for (size_t i = 0; i < Rows; ++i)
                result._data[i] = -_data[i];
            return result;
        }

        tensor<T, Cols>* data() {
            return _data;
        }

        const tensor<T, Cols>* data() const {
            return _data;
        }

        // Matrix operations

        /// \brief Matrix product.
        template <size_t OtherCols>
        typename tensor_t::template convolution_t<1, 0, Cols, OtherCols>
        operator*(const tensor<T, Cols, OtherCols>& other) const {
            return sum<1, 0>(*this, other);
        }

        /// \brief Matrix*vector product.
        tensor<T, Rows>
        operator*(const tensor<T, Cols>& other) const {
            return sum<1, 0>(*this, other);
        }

        /// \brief Directly set a row
        template <size_t I>
        void set_row(const tensor<T, Cols>& row) { set_row(I, row); }
        void set_row(size_t i, const tensor<T, Cols>& row) {
            for (size_t j = 0; j < Cols; ++j)
                _data[i][j] = row[j];
        }

        /// \brief Directly set a column
        template <size_t J>
        void set_col(const tensor<T, Rows>& col) { set_col(J, col); }
        void set_col(size_t j, const tensor<T, Rows>& col) {
            for (size_t i = 0; i < Cols; ++i)
                _data[i][j] = col[i];
        }

        /// \brief Extract a row vector.
        template <size_t I>
        tensor<T, Cols> row() const { return row(I); }
        tensor<T, Cols> row(size_t i) const {
            tensor<T, Cols> result;
            for (size_t j = 0; j < Cols; ++j)
                result[j] = _data[i][j];
            return result;
        }

        /// \brief Extract a column vector.
        template <size_t J>
        tensor<T, Rows> col() const { return col(J); }
        tensor<T, Rows> col(size_t j) const {
            tensor<T, Rows> result;
            for (size_t i = 0; i < Cols; ++i)
                result[i] = _data[i][j];
            return result;
        }

        /// \brief Return the transposed matrix.
        tensor<T, Cols, Rows> transpose() const {
            tensor<T, Cols, Rows> result;
            for (size_t i = 0; i < Rows; ++i)
            for (size_t j = 0; j < Cols; ++j)
                result[j][i] = _data[i][j];
            return result;
        }

        /// \brief Return the minor matrix (with col I and row J removed)
        template<size_t I, size_t J>
        T minor() const {
            return _data[1-I][1-J];
        }

        /// \brief Return a vector containing concatenated rows
        vector<T, 4> rows() const {
            return vector<T, 4>(_data[0][0], _data[0][1], _data[1][0], _data[1][1]);
        }

        /// \brief Return a vector containing concatenated cols
        vector<T, 4> cols() const {
            return vector<T, 4>(_data[0][0], _data[1][0], _data[0][1], _data[1][1]);
        }

        // Square matrix operations

        static constexpr tensor_t identity()
        {
            return tensor_t(tensor<T, 2>(1));
        }

        /// \brief Directly set the diagonal
        void set_diag(const tensor<T, 2>& diag) {
            _data[0][0] = diag[0];
            _data[1][1] = diag[1];
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

    protected:
        data_t _data;
    };

    /// \class tensor<T, N>
    /// \brief Degenerate, rank-one template for the tensor class.
    ///
    /// Higher ranking tensors contain arrays of tensors of one lower rank.
    /// This is the base, first-rank case which contains only an array of T.
    ///
    /// Vector-style methods have been added to this specialization
    template <typename T, size_t N>
    class tensor<T, N>
    {
    public:

        /// \brief Get the number of indexes of the tensor. In this case ::value will always be 1.
        using rank = detail::Rank<N>;

        /// \brief Get the number of dimensions represented by the Ith index of the tensor.
        // In this degenerative case, 'I' should always be zero.
        template <size_t I>
        using dimension = detail::Dimension<I, N>;
        using initializer_t = std::initializer_list<T>;
        using tensor_t = tensor<T, N>;
        using data_t = T[N];
        using element_t = T;

        /// \brief Determine the tensor type which would result from convolution with another tensor.
        template <size_t I, size_t J, size_t... M>
        using convolution_t = typename detail::TensorConvolution< detail::Rank<N>::value, I, J, T, N, M... >::tensor_t;

        tensor() {}

        tensor(const T& value) {
            std::fill(_data, _data + N, value);
        }

        tensor(const initializer_t& values) {
            std::copy(values.begin(), values.end(), _data);
        }

        template <typename OtherT>
        tensor(const tensor<OtherT, N>& other) {
            for (size_t i = 0; i < N; ++i)
                _data[i] = static_cast<T>(other[i]);
        }

        template <typename... Args>
        tensor(Args... args) : _data{ args... } {}

        T& operator[](size_t i) {
            return _data[i];
        }

        const T& operator[](size_t i) const {
            return _data[i];
        }

        bool operator==(const tensor_t& other) const {
            for (size_t i = 0; i < N; ++i)
                if (_data[i] != other._data[i])
                    return false;
            return true;
        }

        bool operator!=(const tensor_t& other) const {
            return !operator==(other);
        }

        tensor_t& operator*=(const T& value) {
            for (size_t i = 0; i < N; ++i)
                _data[i] *= value;
            return *this;
        }

        tensor_t operator*(const T& value) const {
            tensor_t result(*this);
            result *= value;
            return result;
        }

        tensor_t& operator/=(const T& value) {
            for (size_t i = 0; i < N; ++i)
                _data[i] /= value;
            return *this;
        }

        tensor_t operator/(const T& value) const {
            tensor_t result(*this);
            result /= value;
            return result;
        }

        tensor_t& operator+=(const tensor_t& other) {
            for (size_t i = 0; i < N; ++i)
                _data[i] += other._data[i];
            return *this;
        }

        tensor_t operator+(const tensor_t& other) const {
            tensor<T, N> result(*this);
            result += other;
            return result;
        }

        tensor_t& operator-=(const tensor_t& other) {
            for (size_t i = 0; i < N; ++i)
                _data[i] -= other._data[i];
            return *this;
        }

        tensor_t operator-(const tensor_t& other) const {
            tensor_t result(*this);
            result -= other;
            return result;
        }

        tensor_t operator-() const {
            tensor_t result;
            for (size_t i = 0; i < N; ++i)
                result._data[i] = -_data[i];
            return result;
        }

        T* data() {
            return _data;
        }

        const T* data() const {
            return _data;
        }

        // Vector operations

        T magnitude_sq() const {
            return sum<0, 0>(*this, *this);
        }

        T magnitude() const {
            return sqrt(magnitude_sq());
        }

        tensor<T, N>& normalize() {
            T mag_inv = static_cast<T>(1) / magnitude();
            for (size_t i = 0; i < N; ++i)
                _data[i] *= mag_inv;
            return *this;
        }

        tensor<T, N> normal() const {
            tensor<T, N> result(*this);
            result.normalize();
            return result;
        }

        template<size_t I>
        tensor<T, N / 2> half() const {
            tensor<T, N / 2> result;
            for (size_t i = 0; i < N/2; ++i)
                result[i] = _data[i + (I * N / 2)];
            return result;
        }

        void split(tensor<T, N/2>& a, tensor<T, N/2>& b) const {
            static_assert(N%2 == 0, "Vector must have even dimensionality.");
            for (size_t i = 0; i < N/2; ++i) {
                a[i] = _data[i];
                b[i] = _data[i + N/2];
            }
        }

    protected:
        data_t _data;
    };

    template <typename T>
    class tensor<T, 4>
    {
    public:

        static constexpr size_t N = 4;

        /// \brief Get the number of indexes of the tensor. In this case ::value will always be 1.
        using rank = detail::Rank<N>;

        /// \brief Get the number of dimensions represented by the Ith index of the tensor.
        // In this degenerative case, 'I' should always be zero.
        template <size_t I>
        using dimension = detail::Dimension<I, N>;
        using initializer_t = std::initializer_list<T>;
        using tensor_t = tensor<T, N>;
        using element_t = T;

        /// \brief Determine the tensor type which would result from convolution with another tensor.
        template <size_t I, size_t J, size_t... M>
        using convolution_t = typename detail::TensorConvolution< detail::Rank<N>::value, I, J, T, N, M... >::tensor_t;

        tensor() {}

        tensor(const T& value) : x(value), y(value), z(value), w(value) {}

        tensor(const T& x, const T& y, const T& z, const T& w) : x(x), y(y), z(z), w(w) {}

        tensor(const initializer_t& values) {
            auto it = values.begin();
            x = *(it++);
            y = *(it++);
            z = *(it++);
            w = *(it++);
        }

        template <typename OtherT>
        tensor(const tensor<OtherT, N>& other) :
            x(static_cast<T>(other[0])),
            y(static_cast<T>(other[1])),
            z(static_cast<T>(other[2])),
            w(static_cast<T>(other[3])) {}

        T& operator[](size_t i) {
            switch (i)
            {
                case 0: return x;
                case 1: return y;
                case 2: return z;
            }
            return w;
        }

        const T& operator[](size_t i) const {
            switch (i)
            {
                case 0: return x;
                case 1: return y;
                case 2: return z;
            }
            return w;
        }

        bool operator==(const tensor_t& other) const {
            return x == other[0]
                && y == other[1]
                && z == other[2]
                && w == other[3];
        }

        bool operator!=(const tensor_t& other) const {
            return !operator==(other);
        }

        tensor_t& operator*=(const T& value) {
            x *= value;
            y *= value;
            z *= value;
            w *= value;
            return *this;
        }

        tensor_t operator*(const T& value) const {
            tensor_t result(*this);
            result *= value;
            return result;
        }

        tensor_t& operator*=(const tensor_t& other) {
            x *= other[0];
            y *= other[1];
            z *= other[2];
            w *= other[3];
            return *this;
        }

        tensor_t& operator/=(const tensor_t& other) {
            x /= other[0];
            y /= other[1];
            z /= other[2];
            w /= other[3];
            return *this;
        }

        tensor_t operator*(const tensor_t& other) const {
            tensor_t result(*this);
            result *= other;
            return result;
        }

        tensor_t operator/(const tensor_t& other) const {
            tensor_t result(*this);
            result /= other;
            return result;
        }

        tensor_t& operator+=(const tensor_t& other) {
            x += other[0];
            y += other[1];
            z += other[2];
            w += other[3];
            return *this;
        }

        tensor_t operator+(const tensor_t& other) const {
            tensor<T, N> result(*this);
            result += other;
            return result;
        }

        tensor_t& operator-=(const tensor_t& other) {
            x -= other[0];
            y -= other[1];
            z -= other[2];
            w -= other[3];
            return *this;
        }

        tensor_t operator-(const tensor_t& other) const {
            tensor_t result(*this);
            result -= other;
            return result;
        }

        tensor_t operator-() const {
            return tensor_t(-x, -y, -z, -w);
        }

        T* data() {
            return &x;
        }

        const T* data() const {
            return &x;
        }

        // Vector operations

        T magnitude_sq() const {
            return sum<0, 0>(*this, *this);
        }

        T magnitude() const {
            return sqrt(magnitude_sq());
        }

        tensor<T, N>& normalize() {
            T mag_inv = static_cast<T>(1) / magnitude();
            x *= mag_inv;
            y *= mag_inv;
            z *= mag_inv;
            w *= mag_inv;
            return *this;
        }

        tensor<T, N> normal() const {
            tensor<T, N> result(*this);
            result.normalize();
            return result;
        }

        template<size_t I>
        tensor<T, N / 2> half() const {
            tensor<T, N / 2> result;
            for (size_t i = 0; i < N/2; ++i)
                result[i] = operator[](i + (I * N / 2));
            return result;
        }

        void split(tensor<T, N/2>& a, tensor<T, N/2>& b) const {
            static_assert(N%2 == 0, "Vector must have even dimensionality.");
            for (size_t i = 0; i < N/2; ++i) {
                a[i] = operator[](i);
                b[i] = operator[](i + N/2);
            }
        }

    public:

        T x;
        T y;
        T z;
        T w;
    };

    template <typename T>
    class tensor<T, 3>
    {
    public:

        static constexpr size_t N = 3;

        /// \brief Get the number of indexes of the tensor. In this case ::value will always be 1.
        using rank = detail::Rank<N>;

        /// \brief Get the number of dimensions represented by the Ith index of the tensor.
        // In this degenerative case, 'I' should always be zero.
        template <size_t I>
        using dimension = detail::Dimension<I, N>;
        using initializer_t = std::initializer_list<T>;
        using tensor_t = tensor<T, N>;
        using element_t = T;

        /// \brief Determine the tensor type which would result from convolution with another tensor.
        template <size_t I, size_t J, size_t... M>
        using convolution_t = typename detail::TensorConvolution< detail::Rank<N>::value, I, J, T, N, M... >::tensor_t;

        tensor() {}

        tensor(const T& value) : x(value), y(value), z(value) {}

        tensor(const T& x, const T& y, const T& z) : x(x), y(y), z(z) {}

        tensor(const initializer_t& values) {
            auto it = values.begin();
            x = *(it++);
            y = *(it++);
            z = *(it++);
        }

        template <typename OtherT>
        tensor(const tensor<OtherT, N>& other) :
            x(static_cast<T>(other[0])),
            y(static_cast<T>(other[1])),
            z(static_cast<T>(other[2]))
        {}

        T& operator[](size_t i) {
            switch (i)
            {
                case 0: return x;
                case 1: return y;
            }
            return z;
        }

        const T& operator[](size_t i) const {
            switch (i)
            {
                case 0: return x;
                case 1: return y;
            }
            return z;
        }

        bool operator==(const tensor_t& other) const {
            return x == other[0]
                && y == other[1]
                && z == other[2];
        }

        bool operator!=(const tensor_t& other) const {
            return !operator==(other);
        }

        tensor_t& operator*=(const T& value) {
            x *= value;
            y *= value;
            z *= value;
            return *this;
        }

        tensor_t operator*(const T& value) const {
            tensor_t result(*this);
            result *= value;
            return result;
        }

        tensor_t& operator*=(const tensor_t& other) {
            x *= other[0];
            y *= other[1];
            z *= other[2];
            return *this;
        }

        tensor_t& operator/=(const tensor_t& other) {
            x /= other[0];
            y /= other[1];
            z /= other[2];
            return *this;
        }

        tensor_t operator*(const tensor_t& other) const {
            tensor_t result(*this);
            result *= other;
            return result;
        }

        tensor_t operator/(const tensor_t& other) const {
            tensor_t result(*this);
            result /= other;
            return result;
        }

        tensor_t& operator+=(const tensor_t& other) {
            x += other[0];
            y += other[1];
            z += other[2];
            return *this;
        }

        tensor_t operator+(const tensor_t& other) const {
            tensor<T, N> result(*this);
            result += other;
            return result;
        }

        tensor_t& operator-=(const tensor_t& other) {
            x -= other[0];
            y -= other[1];
            z -= other[2];
            return *this;
        }

        tensor_t operator-(const tensor_t& other) const {
            tensor_t result(*this);
            result -= other;
            return result;
        }

        tensor_t operator-() const {
            return tensor_t(-x, -y, -z);
        }

        T* data() {
            return &x;
        }

        const T* data() const {
            return &x;
        }

        // Vector operations

        T magnitude_sq() const {
            return sum<0, 0>(*this, *this);
        }

        T magnitude() const {
            return sqrt(magnitude_sq());
        }

        tensor<T, N>& normalize() {
            T mag_inv = static_cast<T>(1) / magnitude();
            x *= mag_inv;
            y *= mag_inv;
            z *= mag_inv;
            return *this;
        }

        tensor<T, N> normal() const {
            tensor<T, N> result(*this);
            result.normalize();
            return result;
        }

        template<size_t I>
        tensor<T, N / 2> half() const {
            tensor<T, N / 2> result;
            for (size_t i = 0; i < N/2; ++i)
                result[i] = operator[](i + (I * N / 2));
            return result;
        }

    public:

        T x;
        T y;
        T z;
    };

    template <typename T>
    class tensor<T, 2>
    {
    public:

        static constexpr size_t N = 2;

        /// \brief Get the number of indexes of the tensor. In this case ::value will always be 1.
        using rank = detail::Rank<N>;

        /// \brief Get the number of dimensions represented by the Ith index of the tensor.
        // In this degenerative case, 'I' should always be zero.
        template <size_t I>
        using dimension = detail::Dimension<I, N>;
        using initializer_t = std::initializer_list<T>;
        using tensor_t = tensor<T, N>;
        using element_t = T;

        /// \brief Determine the tensor type which would result from convolution with another tensor.
        template <size_t I, size_t J, size_t... M>
        using convolution_t = typename detail::TensorConvolution< detail::Rank<N>::value, I, J, T, N, M... >::tensor_t;

        tensor() {}

        tensor(const T& value) : x(value), y(value) {}

        tensor(const T& x, const T& y) : x(x), y(y) {}

        tensor(const initializer_t& values) {
            auto it = values.begin();
            x = *(it++);
            y = *(it++);
        }

        template <typename OtherT>
        tensor(const tensor<OtherT, N>& other) :
            x(static_cast<T>(other[0])),
            y(static_cast<T>(other[1]))
        {}

        T& operator[](size_t i) {
            return i == 0 ? x : y;
        }

        const T& operator[](size_t i) const {
            return i == 0 ? x : y;
        }

        bool operator==(const tensor_t& other) const {
            return x == other[0]
                && y == other[1];
        }

        bool operator!=(const tensor_t& other) const {
            return !operator==(other);
        }

        tensor_t& operator*=(const T& value) {
            x *= value;
            y *= value;
            return *this;
        }

        tensor_t operator*(const T& value) const {
            tensor_t result(*this);
            result *= value;
            return result;
        }

        tensor_t& operator*=(const tensor_t& other) {
            x *= other[0];
            y *= other[1];
            return *this;
        }

        tensor_t& operator/=(const tensor_t& other) {
            x /= other[0];
            y /= other[1];
            return *this;
        }

        tensor_t operator*(const tensor_t& other) const {
            tensor_t result(*this);
            result *= other;
            return result;
        }

        tensor_t operator/(const tensor_t& other) const {
            tensor_t result(*this);
            result /= other;
            return result;
        }

        tensor_t& operator+=(const tensor_t& other) {
            x += other[0];
            y += other[1];
            return *this;
        }

        tensor_t operator+(const tensor_t& other) const {
            tensor<T, N> result(*this);
            result += other;
            return result;
        }

        tensor_t& operator-=(const tensor_t& other) {
            x -= other[0];
            y -= other[1];
            return *this;
        }

        tensor_t operator-(const tensor_t& other) const {
            tensor_t result(*this);
            result -= other;
            return result;
        }

        tensor_t operator-() const {
            return tensor_t(-x, -y);
        }

        T* data() {
            return &x;
        }

        const T* data() const {
            return &x;
        }

        // Vector operations

        T magnitude_sq() const {
            return sum<0, 0>(*this, *this);
        }

        T magnitude() const {
            return sqrt(magnitude_sq());
        }

        tensor<T, N>& normalize() {
            T mag_inv = static_cast<T>(1) / magnitude();
            x *= mag_inv;
            y *= mag_inv;
            return *this;
        }

        tensor<T, N> normal() const {
            tensor<T, N> result(*this);
            result.normalize();
            return result;
        }

        template<size_t I>
        tensor<T, N / 2> half() const {
            tensor<T, N / 2> result;
            for (size_t i = 0; i < N/2; ++i)
                result[i] = operator[](i + (I * N / 2));
            return result;
        }

        void split(tensor<T, N/2>& a, tensor<T, N/2>& b) const {
            static_assert(N%2 == 0, "Vector must have even dimensionality.");
            for (size_t i = 0; i < N/2; ++i) {
                a[i] = operator[](i);
                b[i] = operator[](i + N/2);
            }
        }

    public:

        T x;
        T y;
    };
}

#endif
