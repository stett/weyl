/// \file weyl.h
/// \brief Weyl is a single-header library for doing math with dense tensors of any finite rank and dimension.
/// \copyright Copyright (c) 2017 Steven Barnett
/// 
/// Weyl is used to perform sums, component products, and inner products between tensors of any
/// rank and dimension. The function weyl::sum is the workhorse of the library, and is used to
/// compute the inner product of two tensors along dimensions of equal length.
/// 
/// Example 1: Vector Dot Product
///
/// The sum across the only index of two single-index tensors, if the indexes are 3-dimensional,
/// is a the dot-product of two 3-dimensional vectors.
///     [ a b c ] * [ d e f ] = (a * d) + (b * e) + (c * f)
/// or
///     weyl::tensor<float, 3> abc({ a, b, c});
///     weyl::tensor<float, 3> def({ d, e, f});
///     float product = weyl::sum<0, 0>(abc, def);
/// 
/// Example 2: Matrix Product
/// 
/// The sum across the second index of a 3x3 tensor with the first index of another 3x3 tensor is essentially a matrix product.
/// 
///     [ a b c ]   [ j k l ]   [ (a*j)+(b*m)+(c*p)  (a*k)+(b*n)+(c*q)  (a*l)+(b*o)+(c*r) ]
///     [ d e f ] * [ m n o ] = [ (d*j)+(e*m)+(f*p)  (d*k)+(e*n)+(f*q)  (d*l)+(e*o)+(f*r) ]
///     [ g h i ]   [ p q r ]   [ (g*j)+(h*m)+(i*p)  (g*k)+(h*n)+(i*q)  (g*l)+(h*o)+(i*r) ]
/// or
///     weyl::tensor<float, 3, 3> a({ { ... }, { ... }, { ... } });
///     weyl::tensor<float, 3, 3> b({ { ... }, { ... }, { ... } });
///     weyl::tensor<float, 3, 3> product = weyl::sum<1, 0>(a, b);
/// 

#pragma once
#ifndef _WEYL_H_
#define _WEYL_H_

#include <initializer_list>
#include <algorithm>

namespace weyl
{
    template <typename T, size_t N0, size_t... N>
    class tensor;

    template <typename, size_t, size_t>
    class matrix;

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

        /// \struct MinorsElement
        /// \brief Minor determinant computation element, for use in a MatrixMap template.
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

        /// \struct CofactorElement
        /// \brief Cofactor computation element, for use in a MatrixMap template.
        template <typename T, size_t Size, size_t i, size_t j>
        struct CofactorElement
        {
            static inline void compute(const matrix<T, Size, Size>& src, matrix<T, Size, Size>& dst) {
                const T sign = static_cast<T>(((long)i + (long)j) % 2 ? -1 : 1);
                const T val = sign * src.minor<i, j>().det();
                dst[i][j] = val;
            }
        };

        /// \struct MatrixMap
        /// \brief Perform a static computation on each of the elements of a matrix.
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

        tensor_t& operator*=(const tensor_t& other) {
            for (size_t i = 0; i < N0; ++i)
                _data[i] *= other._data[i];
            return *this;
        }

        tensor_t operator*(const tensor_t& other) const {
            tensor_t result(*this);
            result *= other;
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

    /// \class tensor<T, N>
    /// \brief Degenerate, rank-one template for the tensor class.
    ///
    /// Higher ranking tensors contain arrays of tensors of one lower rank.
    /// This is the base, first-rank case which contains only an array of T.
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

        template <typename... Args>
        tensor(Args... args) : _data{ args... } {}

        tensor(const tensor_t& other) {
            std::copy(other._data, other._data + N, _data);
        }

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

        tensor_t& operator*=(const tensor_t& other) {
            for (size_t i = 0; i < N; ++i)
                _data[i] *= other._data[i];
            return *this;
        }

        tensor_t operator*(const tensor_t& other) const {
            tensor_t result(*this);
            result *= other;
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

    protected:
        data_t _data;
    };

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
    cat(const tensor<T, N0, M0>& a, const tensor<T, N1, M1>& b) {
        tensor<T, N0 + N1, M0 + M1> result;
        for (size_t i = 0; i < N0; ++i)
        for (size_t j = 0; j < M0; ++j)
            result[i][j] = a[i][j];
        for (size_t i = 0; i < N1; ++i)
        for (size_t j = 0; j < M1; ++j)
            result[N0 + i][M0 + j] = b[i][j];
        return result;
    }


    /// \class vector
    /// \brief Specialization of the tensor class for rank-1 tensors.
    template <typename T, size_t N>
    class vector : public tensor<T, N>
    {
    public:

        using tensor_t = tensor<T, N>;
        using initializer_t = typename tensor_t::initializer_t;

        vector() : tensor_t() {}

        vector(const T& value) : tensor_t(value) { }

        vector(const initializer_t& values) : tensor_t(values) { }

        vector(const tensor_t& other) : tensor_t(other) { }

        vector(const vector<T, N - 1>& other, T value) {
            for (size_t i = 0; i < N - 1; ++i)
                this->_data[i] = other[i];
            this->_data[N - 1] = value;
        }

        vector(const vector<T, N + 1>& other) {
            for (size_t i = 0; i < N; ++i)
                this->_data[i] = other[i];
        }

        T magnitude_sq() const {
            return sum<0, 0>(*this, *this);
        }

        T magnitude() const {
            return sqrt(magnitude_sq());
        }

        template<size_t I>
        vector<T, N / 2> half() const {
            vector<T, N / 2> result;
            for (size_t i = 0; i < N/2; ++i)
                result[i] = this->_data[i + (I * N / 2)];
            return result;
        }

        void split(vector<T, N/2>& a, vector<T, N/2>& b) const {
            static_assert(N%2 == 0, "Vector must have even dimensionality.");
            for (size_t i = 0; i < N/2; ++i) {
                a[i] = this->_data[i];
                b[i] = this->_data[i + N/2];
            }
        }
    };

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


    /// \class matrix
    /// \brief Specialization of the tensor class for rank-2 tensors.
    template <typename T, size_t Rows, size_t Cols>
    class matrix : public tensor<T, Rows, Cols>
    {
    public:

        using tensor_t = tensor<T, Rows, Cols>;

        matrix() : tensor_t() {}

        matrix(const T& value) : tensor_t(static_cast<T>(0)) {
            for (size_t i = 0; i < Rows; ++i)
            for (size_t j = 0; j < Rows; ++j)
                this->_data[i][j] = value;
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
                this->_data[i][j] = row[j];
        }

        /// \brief Directly set a column
        template <size_t J>
        void set_col(const vector<T, Rows>& col) { set_col(J, col); }
        void set_col(size_t j, const vector<T, Rows>& col) {
            for (size_t i = 0; i < Cols; ++i)
                this->_data[i][j] = col[i];
        }

        /// \brief Extract a row vector.
        template <size_t I>
        vector<T, Cols> row() const { return row(I); }
        vector<T, Cols> row(size_t i) const {
            vector<T, Cols> result;
            for (size_t j = 0; j < Cols; ++j)
                result[j] = this->_data[i][j];
            return result;
        }

        /// \brief Extract a column vector.
        template <size_t J>
        vector<T, Rows> col() const { return col(J); }
        vector<T, Rows> col(size_t j) const {
            vector<T, Rows> result;
            for (size_t i = 0; i < Cols; ++i)
                result[i] = this->_data[i][j];
            return result;
        }

        /// \brief Return the transposed matrix.
        matrix<T, Cols, Rows> transpose() const {
            matrix<T, Cols, Rows> result;
            for (size_t i = 0; i < Rows; ++i)
            for (size_t j = 0; j < Cols; ++j)
                result[j][i] = this->_data[i][j];
            return result;
        }

        /// \brief Return the minor matrix (with col I and row J removed)
        template<size_t I, size_t J>
        matrix<T, Rows-1, Cols-1> minor() const {
            matrix<T, Rows-1, Cols-1> m;
            for (size_t i = 0; i < I; ++i)
            for (size_t j = 0; j < J; ++j)
                m[i][j] = this->_data[i][j];
            for (size_t i = I+1; i < Rows; ++i)
            for (size_t j = J+1; j < Cols; ++j)
                m[i-1][j-1] = this->_data[i][j];
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
                this->_data[i][i] = value;
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
                this->_data[i][j] = row[j];
        }

        /// \brief Directly set a column
        template <size_t J>
        void set_col(const vector<T, Size>& col) { set_col(J, col); }
        void set_col(size_t j, const vector<T, Size>& col) {
            for (size_t i = 0; i < Size; ++i)
                this->_data[i][j] = col[i];
        }

        /// \brief Extract a row vector.
        template <size_t I>
        vector<T, Size> row() const { return row(I); }
        vector<T, Size> row(size_t i) const {
            vector<T, Size> result;
            for (size_t j = 0; j < Size; ++j)
                result[j] = this->_data[i][j];
            return result;
        }

        /// \brief Extract a column vector.
        template <size_t J>
        vector<T, Size> col() const { return col(J); }
        vector<T, Size> col(size_t j) const {
            vector<T, Size> result;
            for (size_t i = 0; i < Size; ++i)
                result[i] = this->_data[i][j];
            return result;
        }

        /// \brief Return the transposed matrix.
        matrix<T, Size, Size> transpose() const {
            matrix<T, Size, Size> result;
            for (size_t i = 0; i < Size; ++i)
            for (size_t j = 0; j < Size; ++j)
                result[j][i] = this->_data[i][j];
            return result;
        }

        /// \brief Return the minor matrix (with col I and row J removed)
        template<size_t I, size_t J>
        matrix<T, Size-1, Size-1> minor() const {
            matrix<T, Size-1, Size-1> m;
            for (size_t i = 0; i < Size; ++i)
            for (size_t j = 0; j < Size; ++j)
                if (i != I && j != J)
                    m[i < I ? i : i - 1][j < J ? j : j - 1] = this->_data[i][j];
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
                result += this->_data[i][0] * C[i][0];
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
                this->_data[i][i] = value;
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
                this->_data[i][j] = row[j];
        }

        /// \brief Directly set a column
        template <size_t J>
        void set_col(const vector<T, 2>& col) { set_col(J, col); }
        void set_col(size_t j, const vector<T, 2>& col) {
            for (size_t i = 0; i < 2; ++i)
                this->_data[i][j] = col[i];
        }

        /// \brief Extract a row vector.
        template <size_t I>
        vector<T, 2> row() const { return row(I); }
        vector<T, 2> row(size_t i) const {
            vector<T, 2> result;
            for (size_t j = 0; j < 2; ++j)
                result[j] = this->_data[i][j];
            return result;
        }

        /// \brief Extract a column vector.
        template <size_t J>
        vector<T, 2> col() const { return col(J); }
        vector<T, 2> col(size_t j) const {
            vector<T, 2> result;
            for (size_t i = 0; i < 2; ++i)
                result[i] = this->_data[i][j];
            return result;
        }

        /// \brief Return the transposed matrix.
        matrix<T, 2, 2> transpose() const {
            matrix<T, 2, 2> result;
            for (size_t i = 0; i < 2; ++i)
            for (size_t j = 0; j < 2; ++j)
                result[j][i] = this->_data[i][j];
            return result;
        }

        /// \brief Return the minor matrix (with col I and row J removed)
        template<size_t I, size_t J>
        matrix<T, 1, 1> minor() const {
            matrix<T, 1, 1> m;
            for (size_t i = 0; i < I; ++i)
            for (size_t j = 0; j < J; ++j)
                m[i][j] = this->_data[i][j];
            for (size_t i = I+1; i < 2; ++i)
            for (size_t j = J+1; j < 2; ++j)
                m[i-1][j-1] = this->_data[i][j];
            return m;
        }

        /// \brief Get the determinant of the matrix
        T det() const {
            return (this->_data[0][0] * this->_data[1][1]) - (this->_data[1][0] * this->_data[0][1]);
        }

        matrix<T, 2, 2> minors() const {
            return matrix<T, 2, 2>({
                { this->_data[1][1], this->_data[1][0] },
                { this->_data[0][1], this->_data[0][0] }
            });
        }

        matrix<T, 2, 2> cofactor() const {
            return matrix<T, 2, 2>({
                { this->_data[1][1], -this->_data[1][0] },
                { -this->_data[0][1], this->_data[0][0] }
            });
        }

        matrix<T, 2, 2> adj() const {
            return matrix<T, 2, 2>({
                { this->_data[1][1], -this->_data[0][1] },
                { -this->_data[1][0], this->_data[0][0] }
            });
        }

        /// \brief Return the inverse of the matrix.
        matrix<T, 2, 2> inverse() const {
            T det_inv = static_cast<T>(1.0f) / det();
            return matrix<T, 2, 2>({
                { det_inv * this->_data[1][1], -det_inv * this->_data[0][1] },
                { -det_inv * this->_data[1][0], det_inv * this->_data[0][0] }
            });
        }
    };


    /// \class quaternion
    /// \brief Simple quaternion structure, composed of a scalar and a vector.
    template <typename T>
    class quaternion
    {
    public:
        typedef vector<T, 3> vtype;

    public:
        quaternion() : s(static_cast<T>(1.0)) {}

        quaternion(const quaternion& other) : s(other.s), v(other.v) {}

        quaternion(T s, const vtype& v) : s(s), v(v) {}

        quaternion(const vtype& angles) { from_euler(angles); }

        quaternion(const vtype& a, const vtype& b) { from_rotation(a,b); }

        ~quaternion() {}

        quaternion& operator=(const quaternion& other) {
            s = other.s;
            v = other.v;
            return *this;
        }

        quaternion& operator+=(const quaternion& other) {
            s += other.s;
            v += other.v;
            return *this;
        }

        quaternion& operator-=(const quaternion& other) {
            s -= other.s;
            v -= other.v;
            return *this;
        }

        quaternion& operator*=(const quaternion& other) {
            T s_old = s;
            s = (s * other.s) - dot(v, other.v);
            v = (s_old * other.v) + (v * other.s) + cross(v, other.v);
            return *this;
        }

        quaternion& operator*=(float c) {
            s *= c;
            v *= c;
            return *this;
        }

        operator matrix<T, 3, 3>() const {
            vector<T, 3> v2 = v * v;
            return matrix<T, 3, 3>({
                { 1.0f - 2.0f * (v2[1] + v2[2]), 2.0f * (v[0]*v[1] + s*v[2]), 2.0f * (v[0]*v[2] - s*v[1]) },
                { 2.0f * (v[0]*v[1] - s*v[2]), 1.0f - 2.0f * (v2[0] + v2[2]), 2.0f * (v[1]*v[2] + s*v[0]) },
                { 2.0f * (v[0]*v[2] + s*v[1]), 2.0f * (v[1]*v[2] - s*v[0]), 1.0f - 2.0f * (v2[0] + v2[1]) }
            }).transpose();
        }

        operator matrix<T, 4, 4>() const {
            matrix<T, 3, 3> m33 = operator matrix<T, 3, 3>();
            matrix<T, 4, 4> m44(0.0f);
            for (size_t i = 0; i < 3; ++i)
            for (size_t j = 0; j < 3; ++j)
                m44[i][j] = m33[i][j];
            m44[3][3] = 1.0f;
            return m44;
        }

        bool operator==(const quaternion& other) const { return (s == other.s) && (v == other.v); }

        bool operator!=(const quaternion& other) const { return !operator==(other); }

        void from_euler(const vtype& angles) {
            float c1 = cos(angles[0]*0.5f);
            float s1 = sin(angles[0]*0.5f);
            float c2 = cos(angles[1]*0.5f);
            float s2 = sin(angles[1]*0.5f);
            float c3 = cos(angles[2]*0.5f);
            float s3 = sin(angles[2]*0.5f);
            s     = c1*c2*c3 + s1*s2*s3;
            v[0] = s1*c2*c3 - c1*s2*s3;
            v[1] = c1*s2*c3 + s1*c2*s3;
            v[2] = c1*c2*s3 - s1*s2*c3;

            // Just to be safe...
            normalize();
        }

        static void axis_angle(quaternion& q, const vtype& axis, float angle) {
            float half_angle = angle * 0.5f;
            q.s = cos(half_angle);
            q.v = axis * sin(half_angle);
            q.normalize();
        }

        static void axis_angle(quaternion& q, const vtype& axis) {
            float angle = length(axis);
            q = quaternion<T>::axis_angle(abs(angle) > std::numeric_limits<float>::epsilon() ? axis / angle : vector<T, 3>(0.0f), angle);
        }

        static quaternion axis_angle(const vtype& axis, float angle) {
            quaternion q;
            axis_angle(q, axis, angle);
            return q;
        }

        static quaternion axis_angle(const vtype& axis) {
            quaternion q;
            axis_angle(q, axis);
            return q;
        }

        // Adapted from: http://lolengine.net/blog/2013/09/18/beautiful-maths-quaternion-from-vectors
        void from_rotation(const vtype& a, const vtype& b) {

            float norm_a_norm_b = sqrt(dot(a, a) * dot(b, b));
            if (norm_a_norm_b < std::numeric_limits<float>::epsilon()) {
                v = vector<T, 3>(0.0f, 1.0f, 0.0f);
                s = 0.0f;
                return;
            }
            float cos_theta = dot(a, b) / norm_a_norm_b;
            if (cos_theta < -1.0f + std::numeric_limits<float>::epsilon()) {
                v = vector<T, 3>(0.0f, 1.0f, 0.0f);
                s = 0.0f;
                return;
            }

            s = sqrt(0.5f * (1.f + cos_theta));
            v = cross(a, b) / (norm_a_norm_b * 2.f * s);

            // Unsure that this is necessary...
            normalize();
        }


        quaternion& normalize() {
            T m_inv = 1.0f / magnitude();
            s *= m_inv;
            v *= m_inv;
            return *this;
        }

        quaternion& conjugate() {
            v *= -1.0f;
            return *this;
        }

        quaternion normalized() const {
            quaternion q(*this);
            return q.normalize();
        }

        quaternion conjugated() const {
            quaternion q(*this);
            return q.conjugate();
        }

        T magnitude() const {
            return sqrt(dot(*this, *this));
        }

    public:
        T s;
        vtype v;
    };

    template <typename T>
    T dot(const quaternion<T>& a, const quaternion<T>& b) {
        return (a.s * b.s) + (a.v[0] * b.v[0]) + (a.v[1] * b.v[1]) + (a.v[2] * b.v[2]);
    }

    template <typename T>
    quaternion<T> normalize(quaternion<T> q) {
        return q.normalized();
    }

    template <typename T>
    quaternion<T> operator+(quaternion<T> a, const quaternion<T>& b) {
        return a += b;
    }

    template <typename T>
    quaternion<T> operator-(quaternion<T> a, const quaternion<T>& b) {
        return a -= b;
    }

    template <typename T>
    quaternion<T> operator*(quaternion<T> a, const quaternion<T>& b) {
        return a *= b;
    }

    template <typename T>
    quaternion<T> operator*(quaternion<T> q, T c) {
        return q *= c;
    }

    template <typename T>
    quaternion<T> operator*(T c, quaternion<T> q) {
        return q *= c;
    }

    template <typename T>
    vector<T, 3> operator*(const quaternion<T>& q, const vector<T, 3>& v) {
        return ((q * quaternion<T>(0.0f, v)) * q.conjugated()).v;
    }

    template <typename T>
    matrix<T, 3, 3> operator*(const quaternion<T>& q, const matrix<T, 3, 3>& m) {
        matrix<T, 3, 3> qm(q);
        return (qm * m) * qm.transpose();
    }
}

#endif